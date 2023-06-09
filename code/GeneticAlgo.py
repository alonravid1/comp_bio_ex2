import numpy as np
import re
from collections import Counter

    
class GeneticAlgo:
    
    def __init__(self, replication_rate, cross_over_rate,
                 mutation_rate, mutation_number, gen_size,
                 word_weight, letter_weight, pairs_weight, 
                 swaps, enc_message, word_dict, letter_freq,
                 pair_freq, executor=None, word_eval_func=None):
        """sets all parameters of the algorithm, generates any independent
        variables such as the rng object and solution representation.

        Args:
            replication_rate (float): portion of generation which is replicated as is to the next
            cross_over_rate (float): portion of generation which is crossed over
            mutation_rate (float): chance for a single letter in a solution to be mutated
            mutation_number (int): maximum number of mutations per solution
            gen_size (int): number of soluttions in  a generation
            executor (process pool): a pool of processes for multiprocessing words
            word_eval_func (function): a picklable function which evaluates a word
            word_weight (float): coefficient of word score
            letter_weight (float): coefficient of letter frequency score
            pairs_weight (float): coefficient of letter pairs frequency score
            enc_message (string): an encoded message
            letter_freq (dict): a dictionary of characters and their frequency
            pair_freq (dict): a dictionary of pairs of characters and their frequency
            word_dict (set): a set of valid words
        """
        
        self.alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])
        self.sol_rep = np.arange(26)
        
        self.encoded_message = enc_message
        self.letter_freq = letter_freq
        self.pair_freq = pair_freq
        self.word_dict = word_dict
        self.replication_rate = replication_rate
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate
        self.mutation_number = mutation_number
        
        # make gen size even to make life easier
        self.gen_size = gen_size + (gen_size % 2)
        
        self.rng = np.random.default_rng()
        self.executor = executor
        self.word_eval_func = word_eval_func
        
        self.word_weight = word_weight
        self.letter_weight = letter_weight
        self.pairs_weight = pairs_weight
        self.temp_coeff = 0
        self.fitness_count = 0

        self.replicated_portion = int(self.gen_size*self.replication_rate)
        self.replicated_portion += self.replicated_portion % 2
        self.crossed_over_portion = (self.gen_size - self.replicated_portion)

        # for evaluation, remove all non abc characters
        self.encoded_message = re.sub('[0-9\[\](){}<>;@&^%$!*?,.\n]', '', self.encoded_message)
        self.swaps = swaps
        
    def coverage(self, solution):
        """calculate the coverage of valid words out of the words in a solution

        Args:
            solution (np.array): an array of integers between 0 and 25 representing the alphabet

        Returns:
            float: portion coverage
        """
        cut_message = self.decode_message(solution)
        message_words = cut_message.split(" ")
        count = 0
        total = 0

        # remove any empty words
        for word in message_words:
            if word != " " and word != "":
                total +=1
                if word in self.word_dict[len(word)]:
                    count +=1

        return count/total
                        
        
    def init_run(self, iterations=None):
        """generates a random founder generation, sets all run parameters
        their initial values.

        Args:
            iterations (int, optional): maximum number of iterations to run. Defaults to None,
            which is ignored by the main loop.
        """
        self.solutions = self.get_founder_gen()
        self.previous_best_count = 0
        self.previous_best = self.solutions[0].copy()
        self.iteration = 0
        self.score_stats = np.array([(0,0,0)], dtype=[('max', float), ('avg', float), ('cover', float)])
        self.max_iterations = iterations

    def iterate_step(self):
        """runs a single iteration of the algorithm, evloving the next generation,
        checking whether the algorithm should halt, reset or change phase into the
        final letter optimizations.

        Return(within an array):
            best solution: the best solution of the previous generation
            fitness count: how many evaluation the algorithm has done so far
            score stats: tuple of max score, average score and max coverage in the previous
                        generation
            iteration: current iteration
            finisehd flag: True if the algorithm finished running, Flase otherwise
        """


        if self.previous_best_count > 5 or self.iteration == self.max_iterations:
            return_stats = [self.solutions[0], self.fitness_count, self.score_stats,
                             self.iteration, True]
            return return_stats
        
        self.solutions, avg_score, max_score = self.evolve_new_gen(self.solutions)
        coverage = round(self.coverage(self.solutions[0])*100 ,4)
        new_val = np.array([(round(max_score, 4), round(avg_score, 4), coverage)], dtype=[('max', float), ('avg', float), ('cover', float)])
        self.score_stats = np.append(self.score_stats, new_val)

        if (self.previous_best == self.solutions[0]).all():
            self.previous_best_count += 1
        else:
            self.previous_best = self.solutions[0].copy()
            self.previous_best_count = 0

        
        if self.previous_best_count >= 3:
            self.mutation_number = 1
            self.replication_rate = 0.5
            self.replicated_portion = int(self.gen_size*self.replication_rate)
            self.replicated_portion += self.replicated_portion % 2
            self.crossed_over_portion = (self.gen_size - self.replicated_portion)

        self.iteration += 1

        # nested if to prevent unnecessary coverage computation                    
        if self.previous_best_count >= 5:
            if coverage/100 < 0.5:
                # if the last 10 best solutions has not changed
                # it is a sign of early convergence, and the algorithm will 'reset'.
                self.solutions = self.get_founder_gen()
                self.previous_best_count = 0
                self.previous_best = self.solutions[0].copy()
                return self.solutions[0], -1, new_val, self.iteration, False


        
        
        return_stats = [self.solutions[0], self.fitness_count, new_val,
                            self.iteration, False]       
        return return_stats
            
    
    def validate_solution(self, solution):
        """validate and fix a solution representation after a crossover,
        replacing one instance of a letter appearing twice with a missing letter

        Args:
            solution (np.array): an array of numbers representing a character in the alphabet
        """
        double_letters = []
        missing_letters = []
        counter = list(solution)
        for i in range(26):
            if counter.count(i) == 2:
                double_letters.append(i)

            if counter.count(i) == 0:
                missing_letters.append(i)

        self.rng.shuffle(missing_letters)     

        if len(double_letters) != len(missing_letters):
            print("error in validation")
            exit()
        
        if len(double_letters) == 0:
            return
        
        for i in double_letters:
            is_first = bool(self.rng.integers(2))
            for j in range(26):
                if i == solution[j]:
                    if is_first:
                        solution[j] = missing_letters.pop()
                        break
                    else:
                        is_first = True

    
    def cross_over(self, sol1, sol2):
        """crosses over two solutions at a random point,
        and then validates them. any invalid solution in which
        a placement appears more than one has one of the instances
        replaced with a missing placement.

        Args:
            sol1 (np.array): an array of numbers representing a character in the alphabet
            sol2 (np.array): an array of numbers representing a character in the alphabet

        Returns:
            np.array, np.array: the solutions after crossing over and validation
        """
        # choose random point in the dict
        # to swap the dictionaries, with at least 1 swap
        crossing_point = self.rng.integers(26)

        temp = sol1[:crossing_point].copy()
        sol1[:crossing_point], sol2[:crossing_point] = sol2[:crossing_point], temp

        sol1 = sol1.flatten()
        sol2 = sol2.flatten()

        # uniform crossover
        # prob_matrix = self.rng.integers(0, 2, size=26)
        # offspring1 = np.where(prob_matrix == 0, sol1, sol2)
        # offspring2 = np.where(prob_matrix == 1, sol1, sol2)
        
        self.validate_solution(sol1)
        self.validate_solution(sol2)

        return sol1, sol2

    
    def decode_message(self, solution, message=None):
        """apply permutation to message according to the
        solution

        Args:
            solution (np.array): an array of integers between 0 and 25 representing the alphabet
            message (string, optional): message to be decoded, defaults to the class's message

        Returns:
            string: message after applying the solution
        """
        table = str.maketrans("".join(self.alphabet), "".join(self.alphabet[solution]))
        if message != None:
            new_message = message.translate(table)
        else:
            new_message = self.encoded_message.translate(table)
        
        return new_message

    def count_pairs(self, string):
        """Counts the number of pairs of letters in a string and outputs them as a dictionary.

        Args:
            string: The string to count pairs of letters in.

        Returns:
            A dictionary that maps each pair of letters to its count.
        """
        
        # create an empty dictionary to store the counts of each pair of letters.
        pairs_count = {}
        for char1 in self.alphabet:
            for char2 in self.alphabet:
                pairs_count[char1 + char2] = string.count(char1 + char2)

        return pairs_count
        
    def eval_func(self, solution):
        """evaluation function for solutions, calculating the number
        of valid words in it, the letter frequencies and pairs of letters
        frequencies.

        Args:
            solution (np.array): an array of integers between 0 and 25 representing the alphabet

        Returns:
            float: a score for the given solution
        """
        self.fitness_count += 1
        cut_message = self.decode_message(solution)

        spaces = cut_message.count(" ")
        length = len(cut_message) - spaces
        message_words = cut_message.split(" ")

        # remove any empty words
        while message_words.count("") != 0:
            message_words.remove("")
        
        # get dictionary of words and their count
        word_counter = Counter(message_words)
        words = list(word_counter.keys())
        # calculate score once per word
        iterable_args = [[word, self.word_dict[len(word)]] for word in words]
        
        # used eval word to multithread the word score calculation
        score_futures = self.executor.map_async(self.word_eval_func,
                                                iterable_args, chunksize=10)

        word_score = 0
        word_scores = score_futures.get()
        
        # calculate total score with regard to unique word score and its count
        for score, word in word_scores:
            word_score += word_counter[word]*score

        letter_count = Counter(cut_message)
        pair_count = self.count_pairs(cut_message)
        word_score = word_score / len(message_words)
        letter_score = 0
        pair_score = 0

        for char1 in self.alphabet:
            letter_score += (letter_count[char1]/length - self.letter_freq[char1])**2/len(self.alphabet)
            for char2 in self.alphabet:
                pair_score += (pair_count[char1+char2]/(length-1) - self.pair_freq[char1+char2])**2/(len(self.alphabet))**2
               
        score = (self.word_weight*word_score +
                self.letter_weight*(1 - letter_score) + self.pairs_weight*(1 - pair_score))

        return score
    
    
    def mutate(self, solution):
        """for each letter in the solution, apply a random
        chance to mutate and swap it with another.

        Args:
            solution (np.array): an array of integers between 0 and 25 representing the alphabet
        """

        for i in range(self.mutation_number):
            rand = self.rng.random(1)
            if rand <= self.mutation_rate:
                swap1 = self.rng.integers(26)
                swap2 = self.rng.integers(26)
                solution[swap1], solution[swap2] = solution[swap2], solution[swap1]

        return solution

            
    def get_founder_gen(self):
        """generate random solutions by shuffling representation arrays

        Returns:
            np.array: array of arrays representing the alphabet
        """
        # solutions = np(self.sol_rep, (self.gen_size, 1))
        # for i in solutions:
        #     # this is done in a for loop because shuffling all
        #     # at once shuffles them the same way
        #     self.rng.shuffle(i)

        solutions = np.array([self.rng.permutation(26) for i in range(self.gen_size)])
        return solutions
    
    def evolve_new_gen(self, solutions):
        """calculate the score of the previous gen, sort its solutions
        by it from low to high, normalize the score to be between 0 and 1
        and calculate each solution's cumulative score in their sorted order.

        that way each between any 2 solutions will be a probabilty density
        equal to the second solution's normalized score, which will then
        be used to pick solutions for crossover via linear sampling.

        then the function replicates the best solutions into the next generation
        according to the replictaion rate parameter, and the rest are created
        two at a time, from a cross over of by two solutions sampled lineary to
        their score, meaning better solutions will have better chances of being
        crossed over.

        the function then mutates all solutions except for the very best one.

        Args:
            solutions (np.array): array of np arrays, a generation of solutions

        Returns:
            np.array: array containing the new generation
            float: average score of the previous generation,
            float: maximum score of the previous generation
        """
        score_index_arr = np.array([(0, 0) for i in range(self.gen_size)], dtype=[('score', float), ('index', int)])
        
        for index in range(self.gen_size):
            score = self.eval_func(solutions[index])
            score_index_arr[index]['index'] = index
            score_index_arr[index]['score'] = score

        # sorts the solutions in ascending order
        score_index_arr.sort(order='score')
        # get statistics
        max_score = score_index_arr[-1]['score']
        avg_score = np.mean(score_index_arr['score'])
        
        # turn score into fraction of total scores, for linear sampling
        score_sum = np.sum(score_index_arr['score'])
        score_index_arr['score'] = score_index_arr['score'] / score_sum
        
        # make score colmutive, so sampling can be done with a random number between 0 and 1
        for i in range(1, self.gen_size):
            score_index_arr[i]['score'] = score_index_arr[i]['score'] + score_index_arr[i-1]['score']

        # fixing the last score which is sometimes a tiny error bellow 1
        score_index_arr[-1]['score'] = 1

        # replicate the best solutions
        new_solutions = np.zeros((self.gen_size, 26), dtype=int)
        new_solutions[0] = solutions[score_index_arr[-1]['index']].copy()

        for i in range(1, self.replicated_portion):
            new_solutions[i] = solutions[score_index_arr[-1]['index']].copy()
            new_solutions[i] = self.mutate(new_solutions[i])
        
        random_portions = self.rng.random(self.crossed_over_portion)
                
        for i in range(0, self.crossed_over_portion, 2):
            score_rank1 = np.searchsorted(score_index_arr['score'], random_portions[i], side='right')
            score_rank2 = np.searchsorted(score_index_arr['score'], random_portions[i+1], side='right')
            index1 = score_index_arr[score_rank1]['index']
            index2 = score_index_arr[score_rank2]['index']

            sol1, sol2 = self.cross_over(solutions[index1].copy(), solutions[index2].copy())

            sol1 = self.mutate(sol1)
            sol2 = self.mutate(sol2)

            new_sol_index = self.replicated_portion + i
            new_solutions[new_sol_index] = sol1.copy()
            new_solutions[new_sol_index + 1] = sol2.copy()

        return new_solutions, avg_score, max_score