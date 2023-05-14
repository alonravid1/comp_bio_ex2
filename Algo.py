import numpy as np
import re
from timebudget import timebudget
import concurrent.futures
import functools
import time

class Algo:
    
    def __init__(self, enc_message, letter_freq, pair_freq, dict_words,
                  replication_rate, cross_over_rate, mutation_rate, gen_size):
        self.alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])
        self.sol_rep = np.arange(26)
        self.encoded_message = enc_message
        self.letter_freq = letter_freq
        self.pair_freq = pair_freq
        self.dict_words = dict_words
        self.replication_rate = replication_rate
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate
        # make gen size even to make life easier
        self.gen_size = gen_size + (gen_size % 2)
        self.rng = np.random.default_rng(7)
        

    def run(self, iterations):
        solutions = self.get_founder_gen()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
        for i in range(iterations):
            solutions = self.evolve_new_gen(solutions)
        self.executor.shutdown(wait=True)
        return solutions
            
    
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
            print("solution:")
            for i in range(len(solution)):
                print(solution[i])
            print("double_letters:")
            for i in range(len(double_letters)):
                print(double_letters[i])
            print("missing_letters:")
            for i in range(len(missing_letters)):
                print(missing_letters[i])
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
        permutation dictionary

        Args:
            message (string): message to be permutated
            perm_dict (dict): dictionary of permutations

        Returns:
            string: message after applying permutation dictionary
        """

        table = str.maketrans("".join(self.alphabet), "".join(self.alphabet[solution]))
        new_message = message.translate(table)
        
        return new_message
    
    def eval_func(self, solutions, index):
        
        # for evaluation, remove all non abc characters
        cut_message = re.sub('[0-9\[\](){}<>;@&^%$!*?,.\n]', '', self.encoded_message)
        decrypt_message = self.decode_message(cut_message, solutions[index])

        message_words = decrypt_message.split(" ")

        # remove any empty words
        while message_words.count("") != 0:
            message_words.remove("")

        valid_word_count = 0
        letter_score = 0
        pair_score = 0

        for word in message_words:
            if word in self.dict_words:
                # criteriea 1: words in dict/words in message
                valid_word_count += 1
            for i in range(len(word)):
                # criteriea 2: sum of frequencies of single letters
                letter_score += self.letter_freq[word[i]]
                if i+1 < len(word):
                    # criteriea 3: sum of frequencies of pairs of letters
                    pair_score += self.pair_freq[word[i:i+2]]
        
        valid_word_count = valid_word_count / len(message_words)
        letter_score = letter_score / len(decrypt_message)
        pair_score = pair_score / len(decrypt_message)
        score = 10*valid_word_count + 5*letter_score + pair_score
        return score, index
    
    
    def mutate(self, solution):
        for i in range(26):
            rand = self.rng.random(1)
            if rand <= self.mutation_rate:
                swap = self.rng.integers(25)
                
                temp = solution[i].copy()
                solution[i] = solution[swap]
                solution[swap] = temp

            
    def get_founder_gen(self):
        solutions = np.tile(self.sol_rep, (self.gen_size, 1))
        for i in solutions:
            # this is done in a for loop because shuffling all
            # at once shuffles them the same way
            self.rng.shuffle(i)
        return solutions
    
    def softmax(self, x, i, vec_sum, max_val):
        """softmax function which replaces the given value with
        the softmax output value. reducing the max element from each
        element is done for better numerical stabilty.

        Args:
            x (_type_): _description_
        """
        vec = np.exp(x - max_val)
        x = (vec / vec_sum)
        return x, i
    
    def get_index(self, rands, score_index_arr):
        rand1, rand2 = rands[0], rands[1]
        score_rank1 = np.searchsorted(score_index_arr['score'], rand1, side='right')
        score_rank2 = np.searchsorted(score_index_arr['score'], rand2, side='right')
        if score_rank1 == 100 or score_rank2 == 100:
            print("problem!")
            print(rand1)
            print(rand2)
            print(score_index_arr[0])
            print(score_index_arr[-2])
            print(score_index_arr[-1])
        return score_rank1, score_rank2
        
    
    def evolve_new_gen(self, solutions):
        
        dtype = [('score', float), ('index', int)]
        score_index_arr = np.array([(0, 0) for i in range(self.gen_size)], dtype=dtype)
        start1 = time.time()
        # create_si_arr = functools.partial(self.eval_func, score_index_arr=score_index_arr)
        score_futures = [self.executor.submit(self.eval_func,
                                solutions, i) for i in range(self.gen_size)]
        
        for future in concurrent.futures.as_completed(score_futures):
            score, index = future.result()
            score_index_arr[index]['index'] = index
            score_index_arr[index]['score'] = score
        end1 = time.time()
        # concurrent.futures.wait(si_array_futures, return_when="ALL_COMPLETED")
            
        score_sum = np.sum(score_index_arr['score'])
        max_val = np.max(score_index_arr['score'])

        # fixed_softmax = functools.partial(self.softmax, vec_sum=score_sum, max_val=max_val)
        # self.executor.map(fixed_softmax, score_index_arr['score'], chunksize=15)
        start2 = time.time()
        softmax_futures = [self.executor.submit(self.softmax,
                                score_index_arr[i]['score'], i, score_sum, max_val)
                                  for i in range(self.gen_size)]

        for future in concurrent.futures.as_completed(softmax_futures):
            score, index = future.result()
            score_index_arr[index]['score'] = score
        end2 = time.time()
        # concurrent.futures.wait(softmax_futures, return_when="ALL_COMPLETED")

        # sorts the solutions in ascending order
        score_index_arr.sort(order='score')

        # make score colmutive
        for i in range(1, self.gen_size):
            score_index_arr[i]['score'] = score_index_arr[i]['score'] + score_index_arr[i-1]['score']

        # new_solutions = np.zeros((self.gen_size, 26), dtype=int)

        # replicate the best solution
        replicated = int(self.gen_size*self.replication_rate)
        replicated += replicated % 2
        new_solutions = np.tile(solutions[score_index_arr[-1]['index']], (replicated,1))
        crossed_over_portion = (self.gen_size - replicated)//2
        
        random_portions = self.rng.random((crossed_over_portion,2))

        # fixed_get_index = functools.partial(self.get_index, score_index_arr=score_index_arr)
        # cross_over_pairs = self.executor.map(fixed_get_index, random_portions, chunksize=1)
        start3 = time.time()
        cross_over_futures = [self.executor.submit(self.get_index, rands, score_index_arr)
                               for rands in random_portions]
        end3 = time.time()
        start4 = time.time()
        for future in concurrent.futures.as_completed(cross_over_futures):
            score_rank1, score_rank2 = future.result()

            # for all other solutions, pick two at a time to
            # crossover and add to the new generation, biased
            # such that higher scoring solutions have a higher
            # of being picked
            index1 = score_index_arr[score_rank1]['index']
            index2 = score_index_arr[score_rank2]['index']


            sol1, sol2 = self.cross_over(solutions[index1].copy(), solutions[index2].copy())
            new_solutions = np.concatenate((new_solutions, [sol1]), axis=0)
            new_solutions = np.concatenate((new_solutions, [sol2]), axis=0)

        end4 = time.time()
        # self.executor.map(self.mutate, new_solutions, score_index_arr,chunksize=1)

        print(end1-start1)
        print(end2-start2)
        print(end3-start3)
        print(end4-start4)
        return(new_solutions)
        
        


            
