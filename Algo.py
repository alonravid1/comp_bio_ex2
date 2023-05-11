import numpy as np
import re
from timebudget import timebudget
import concurrent.futures

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
        

    def run(self, iterations):
        solutions = self.get_founder_gen()
        for i in range(iterations):
            solutions = self.evolve_new_gen(solutions)

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
            print("error in validation")
            exit()
        
        if len(double_letters) == 0:
            return
        
        for i in double_letters :
            for j in range(26):
                if i == solution[j]:
                    solution[j] = missing_letters.pop()
                    break

    
    def cross_over(self, sol1, sol2):
        # choose random point in the dict
        # to swap the dictionaries, with at least 1 swap
        crossing_point = np.random.randint(1,25)

        temp = sol1[:crossing_point].copy()
        sol1[:crossing_point], sol2[:crossing_point] = sol2[:crossing_point], temp

        sol1 = sol1.flatten()
        sol2 = sol2.flatten()
        
        self.validate_solution(sol1)
        self.validate_solution(sol2)

    
    def apply_perm(self, message, solution):
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
    
    def eval_func(self, solution):
        
        # for evaluation, remove all non abc characters
        cut_message = re.sub('[0-9\[\](){}<>;@&^%$!*?,.\n]', '', self.encoded_message)
        decrypt_message = self.apply_perm(cut_message, solution)

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
        
        # valid_word_count = valid_word_count / len(message_words)
        valid_word_count = valid_word_count / len(message_words)
        letter_score = letter_score / len(decrypt_message)
        pair_score = pair_score / len(decrypt_message)
        score = 10*valid_word_count + 5*letter_score + pair_score
        
        return score
    
    
    
    def mutate(self, solution):
        for i in range(26):
            rand = np.random.rand(1)
            if rand <= self.mutation_rate:
                swap = np.random.randint(25)
                
                temp = solution[i].copy()
                solution[i] = solution[swap]
                solution[swap] = temp

            
    def get_founder_gen(self):
        rng = np.random.default_rng(7)
        solutions = np.tile(self.sol_rep, (self.gen_size, 1))
        for i in solutions:
            # this is done in a for loop because shuffling all
            # at once shuffles them the same way
            rng.shuffle(i)
        return solutions
    
    def softmax(self, x):
        """softmax function, reducing the max element from each
        element for better numerical stabilty

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        vec = np.exp(x - np.max(x))
        vec_sum = np.sum(vec)
        return (vec / vec_sum)
        
    
    @timebudget
    def evolve_new_gen(self, solutions):
        dtype = [('score', float), ('index', int)]
        score_index_arr = np.array([(0, 0) for i in range(self.gen_size)], dtype=dtype)
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            for ind, score in zip(np.arange(self.gen_size), executor.map(self.eval_func, solutions)):
                score = self.softmax(score)
                score_index_arr[ind]['score'] = score
                score_index_arr[ind]['index'] = ind
        
        # sorts the solutions in ascending order
        score_index_arr.sort(order='score')

        # make score colmutive
        for i in range(1, self.gen_size):
            score_index_arr[i]['score'] = score_index_arr[i]['score'] + score_index_arr[i-1]['score']
        new_solutions = np.zeros((self.gen_size, 26), dtype=int)

        # replicate the best solution
        replicated = int(self.gen_size*self.replication_rate)
        replicated += replicated % 2
        new_solutions[0:replicated] = np.tile(solutions[score_index_arr[-1]['index']], (replicated,1))

        for i in range(replicated, self.gen_size, 2):
            # for all other solutions, pick two at a time to
            # crossover and add to the new generation, biased
            # such that higher scoring solutions have a higher
            # of being picked
            rand1 = np.random.rand(1)
            rand2 = np.random.rand(1)

            # rand1 = np.random.randint(0, self.gen_size)
            # rand2 = np.random.randint(0, self.gen_size)

            score_rank1 = np.searchsorted(score_index_arr['score'], rand1, side='right')
            score_rank2 = np.searchsorted(score_index_arr['score'], rand2, side='right')
            index1 = score_index_arr[score_rank1]['index']
            index2 = score_index_arr[score_rank2]['index']

            # print(f"scores1: {score_index_arr['score']}\n rand1: {rand1}")
            # print(f"scores2: {score_index_arr['score']}\n rand2: {rand1}")

            # print(solutions[index1])
            # print(solutions[index2])

            self.cross_over(solutions[index1], solutions[index2])
            
            new_solutions[i] = solutions[index1]
            new_solutions[i+1] = solutions[index2]
            

        for solution in new_solutions:
            self.mutate(solution)

        return(new_solutions)
        
        


            
