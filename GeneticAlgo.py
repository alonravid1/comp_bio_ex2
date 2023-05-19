import numpy as np
import re


    
class GeneticAlgo:
    
    def __init__(self, enc_message, letter_freq, pair_freq,
                 word_set, replication_rate, cross_over_rate,
                 mutation_rate, gen_size, executor, word_eval_func,
                 word_coeff, letter_coeff, pairs_coeff):
        
        self.alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])
        self.sol_rep = np.arange(26)
        
        self.encoded_message = enc_message
        self.letter_freq = letter_freq
        self.pair_freq = pair_freq
        self.word_set = word_set
        self.replication_rate = replication_rate
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate
        
        # make gen size even to make life easier
        self.gen_size = gen_size + (gen_size % 2)
        
        self.rng = np.random.default_rng(7)
        self.executor = executor
        self.word_eval_func = word_eval_func
        
        self.word_coeff = word_coeff
        self.letter_coeff = letter_coeff
        self.pairs_coeff = pairs_coeff
        self.fitness_count = 0

        self.replicated_portion = int(self.gen_size*self.replication_rate)
        self.replicated_portion += self.replicated_portion % 2
        self.crossed_over_portion = (self.gen_size - self.replicated_portion)//2

        # for evaluation, remove all non abc characters
        self.encoded_message = re.sub('[0-9\[\](){}<>;@&^%$!*?,.\n]', '', self.encoded_message)
        
    def coverage(self, solution):
        decrypt_message = self.decode_message(self.encoded_message, solution)
        message_words = decrypt_message.split(" ")
        count = 0
        total = 0

        # remove any empty words
        for word in message_words:
            if word != " ":
                total +=1
                if word in self.word_set:
                    count +=1

        return count/total


    def run(self, iterations=None):
        """starts running the algorithm, with the given number of iterations

        Args:
            iterations (int): number of iterations

        Returns:
            list: a list of solutions, sorted in ascending order
        """
        solutions = self.get_founder_gen()
        previous_best_count = 0
        previous_best = solutions[0].copy()
        i = 0
        while previous_best_count < 10 and i != iterations:
            solutions = self.evolve_new_gen(solutions)
            if (previous_best == solutions[0]).all():
                previous_best_count += 1
            else:
                previous_best = solutions[0].copy()
                previous_best_count = 0
            i += 1

            if previous_best_count >= 5:
                # nested if to prevent unnecessary coverage computation
                if self.coverage(solutions[0]) < 0.6:
                    # if the last 5 best solutions has not changed
                    # and there are still many words that are not in the dictionary
                    # it is a sign of early convergence, and the algorithm will 'reset'.
                    solutions = self.get_founder_gen()
                    previous_best_count = 0
                    previous_best = solutions[0].copy()
                elif self.coverage(solutions[0]) < 0.9:
                    # if the best solution hasnt improved for the past 5
                    # generations but there is only a small portion
                    # of invalid words, increase mutation rate to overcome
                    # local maxima near the global one, and give pair frequency
                    # a higher weight.
                    previous_best_count = 0
                    self.mutation_rate += 0.03
                    self.pairs_coeff = 15
                    self.letter_coeff = 1

                    print(f"changed mode at iteration {i}")
                    
            if i%20 == 0:
                print(f"iteration {i}, score {self.eval_func(solutions[0])}:")
                print(self.decode_message(self.encoded_message, solutions[0])[:100])

        print(f"best solution found at generation {i}, in {self.fitness_count} evaluations")
        print(self.decode_message(self.encoded_message, solutions[0]))
        return solutions[0], self.fitness_count
            
    
    def validate_solution(self, solution):
        """validate and fix a solution representation after a crossover,
        replacing one instance of a letter appearing twice with a missing letter

        Args:
            solution (solution): solution representation
        """

        double_letters = []
        missing_letters = []
        counter = list(solution)
        for i in range(26):
            if counter.count(i) == 2:
                double_letters.append(i)

            if counter.count(i) == 0:
                missing_letters.append(i)


        if len(double_letters) != len(missing_letters):
            print("error in validation")
            exit()
        
        if len(double_letters) == 0:
            return
        
        for i in double_letters:
            for j in range(26):
                if i == solution[j]:
                    solution[j] = missing_letters.pop()
                    break

    
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
        crossing_point = self.rng.integers(1,25)

        temp = sol1[:crossing_point].copy()
        sol1[:crossing_point], sol2[:crossing_point] = sol2[:crossing_point], temp

        sol1 = sol1.flatten()
        sol2 = sol2.flatten()

        self.validate_solution(sol1)
        self.validate_solution(sol2)

        return sol1, sol2

    
    def decode_message(self, message, solution):
        """apply permutation to message according to the
        solution

        Args:
            message (string): message to be decoded
            solution (np.array): an array of integers between 0 and 25 representing the alphabet

        Returns:
            string: message after applying the solution
        """
        table = str.maketrans("".join(self.alphabet), "".join(self.alphabet[solution]))
        new_message = message.translate(table)
        
        return new_message

    
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
        decrypt_message = self.decode_message(self.encoded_message, solution)

        message_words = decrypt_message.split(" ")

        # remove any empty words
        while message_words.count("") != 0:
            message_words.remove("")

        iterable_args = [[word, self.word_set, self.letter_freq, self.pair_freq,
                          self.word_coeff, self.letter_coeff, self.pairs_coeff]
                                                    for word in message_words]

        ## used eval word to multithread this part
        score_futures = self.executor.map_async(self.word_eval_func,
                                                iterable_args, chunksize=500)

        word_scores = np.array(score_futures.get())
        score = np.sum(word_scores)

        return score
    
    
    def mutate(self, solution):
        """_summary_

        Args:
            solution (_type_): _description_
        """
        for i in range(26):
            rand = self.rng.random(1)
            if rand <= self.mutation_rate:
                swap = self.rng.integers(25)
                
                temp = solution[i].copy()
                solution[i] = solution[swap]
                solution[swap] = temp

            
    def get_founder_gen(self):
        """generate random solutions by shuffling representation arrays

        Returns:
            np.array: array of arrays representing the alphabet
        """
        solutions = np.tile(self.sol_rep, (self.gen_size, 1))
        for i in solutions:
            # this is done in a for loop because shuffling all
            # at once shuffles them the same way
            self.rng.shuffle(i)
        return solutions
    
    
    def get_index(self, rands, score_index_arr):
        """_summary_

        Args:
            rands (_type_): _description_
            score_index_arr (_type_): _description_

        Returns:
            _type_: _description_
        """
        rand1, rand2 = rands[0], rands[1]
        score_rank1 = np.searchsorted(score_index_arr['score'], rand1, side='right')
        score_rank2 = np.searchsorted(score_index_arr['score'], rand2, side='right')

        return score_rank1, score_rank2
    
    def evolve_new_gen(self, solutions):
        """_summary_

        Args:
            solutions (_type_): _description_
        """
        score_index_arr = np.array([(0, 0) for i in range(self.gen_size)], dtype=[('score', float), ('index', int)])

        for index in range(self.gen_size):
            score = self.eval_func(solutions[index])
            score_index_arr[index]['index'] = index
            score_index_arr[index]['score'] = score
            
        # turn score into fraction of total scores, for linear sampling
        score_sum = np.sum(score_index_arr['score'])
        score_index_arr['score'] = score_index_arr['score'] / score_sum

        # sorts the solutions in ascending order
        score_index_arr.sort(order='score')
        
        # make score colmutive, so sampling can be done with a random number between 0 and 1
        for i in range(1, self.gen_size):
            score_index_arr[i]['score'] = score_index_arr[i]['score'] + score_index_arr[i-1]['score']

        # fixing the last score which is sometimes a tiny error bellow 1
        score_index_arr[-1]['score'] = 1

        # replicate the best solutions
        best_solutions_indices = [score_index_arr[-1]['index'] for i in range(1, self.replicated_portion+1)]
        new_solutions = solutions[best_solutions_indices].copy()
        
        random_portions = self.rng.random((self.crossed_over_portion,2))
        cross_over_pairs = [self.get_index(rands, score_index_arr) for rands in random_portions]
        
        for pair in cross_over_pairs:
            score_rank1, score_rank2 = pair

            # for all other solutions, pick two at a time to
            # crossover and add to the new generation, biased
            # such that higher scoring solutions have a higher
            # of being picked
            index1 = score_index_arr[score_rank1]['index']
            index2 = score_index_arr[score_rank2]['index']

            sol1, sol2 = self.cross_over(solutions[index1].copy(), solutions[index2].copy())
            self.mutate(sol1)
            self.mutate(sol2)
            new_solutions = np.concatenate((new_solutions, [sol1]), axis=0)
            new_solutions = np.concatenate((new_solutions, [sol2]), axis=0)



        return new_solutions
        
        


            
