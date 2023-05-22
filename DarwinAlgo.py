from GeneticAlgo import GeneticAlgo
import numpy as np

class DarwinAlgo(GeneticAlgo):
    
    def __init__(self, enc_message, letter_freq, pair_freq,
                 word_set, replication_rate, cross_over_rate,
                 mutation_rate, gen_size, executor, word_eval_func,
                 word_coeff, letter_coeff, pairs_coeff, swaps):
        """sets all parameters of the algorithm, generates any independent
        variables such as the rng object and solution representation.

        Args:
            enc_message (string): an encoded message
            letter_freq (dict): a dictionary of characters and their frequency
            pair_freq (dict): a dictionary of pairs of characters and their frequency
            word_set (set): a set of valid words
            replication_rate (float): portion of generation which is replicated as is to the next
            cross_over_rate (float): portion of generation which is crossed over
            mutation_rate (float): chance for a single letter in a solution to be mutated
            gen_size (int): number of soluttions in  a generation
            executor (process pool): a pool of processes for multiprocessing words
            word_eval_func (function): a picklable function which evaluates a word
            word_coeff (float): coefficient of word score
            letter_coeff (float): coefficient of letter frequency score
            pairs_coeff (float): coefficient of letter pairs frequency score
            swaps (int): how many swaps the optimization function does
        """
        
        super().__init__(enc_message, letter_freq, pair_freq,
                         word_set, replication_rate, cross_over_rate,
                         mutation_rate, gen_size, executor, word_eval_func,
                         word_coeff, letter_coeff, pairs_coeff)
        
        self.swaps = swaps
        
        
    def optimization_score(self, solution):
        score = self.eval_func(solution)
        for i in range(self.swaps):
            swap_pos_1 = self.rng.integers(25)
            swap_pos_2 = self.rng.integers(25)
            
            while swap_pos_1 == swap_pos_2:
                # prevent swapping of the same place with itself
                swap_pos_2 = self.rng.integers(25)
            temp = solution[swap_pos_1]
            solution[swap_pos_1] = solution[swap_pos_2]
            solution[swap_pos_2] = temp
            
            new_score = self.eval_func(solution)
            if new_score > score:
                score = new_score
        return score
    
    def evolve_new_gen(self, solutions):
        """_summary_

        Args:
            solutions (_type_): _description_
        """
        score_index_arr = np.array([(0, 0) for i in range(self.gen_size)], dtype=[('score', float), ('index', int)])

        for index in range(self.gen_size):
            score = self.optimization_score(solutions[index].copy())
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

        return new_solutions, avg_score, max_score