from GeneticAlgo import GeneticAlgo
import numpy as np

class LamarckAlgo(GeneticAlgo): 
        
    def optimize_solution(self, solution):
        """function opitmizes a given solution by randomly swapping
        letter in it, and calculating its new score. if higher, it then applies
        the change.
        

        Args:
            solution (np.array): an array of numbers representing a character in the alphabet

        Returns:
            score: the new score of the modified solution
        """
        score = self.eval_func(solution)
        for i in range(self.swaps):
            temp_solution = solution.copy()
            swap_pos_1 = self.rng.integers(26)
            swap_pos_2 = self.rng.integers(26)
            
            while swap_pos_1 == swap_pos_2:
                # prevent swapping of the same place with itself
                swap_pos_2 = self.rng.integers(26)

            temp_solution[swap_pos_1], temp_solution[swap_pos_2] = temp_solution[swap_pos_2], temp_solution[swap_pos_1]
            
            new_score = self.eval_func(temp_solution)
            if new_score > score:
                score = new_score
                solution = temp_solution.copy()
                
        return score, solution
    
    def evolve_new_gen(self, solutions):
        """_summary_

        Args:
            solutions (_type_): _description_
        """
        score_index_arr = np.array([(0, 0) for i in range(self.gen_size)], dtype=[('score', float), ('index', int)])

        for index in range(self.gen_size):
            score, solutions[index] = self.optimize_solution(solutions[index])
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