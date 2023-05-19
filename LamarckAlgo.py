from GeneticAlgo import GeneticAlgo

class LamarckAlgo(GeneticAlgo):
    
    def __init__(self, enc_message, letter_freq, pair_freq,
                 word_set, replication_rate, cross_over_rate,
                 mutation_rate, gen_size, executor, word_eval_func,
                 word_coeff, letter_coeff, pairs_coeff, swaps):
        
        super().__init__(enc_message, letter_freq, pair_freq,
                         word_set, replication_rate, cross_over_rate,
                         mutation_rate, gen_size, executor, word_eval_func,
                         word_coeff, letter_coeff, pairs_coeff)
        
        self.swaps = swaps
        
        
    def optimize_solution(self, solution):
        for i in range(self.swaps):
            swap_pos_1 = self.rng.integers(25)
            swap_pos_2 = self.rng.integers(25)
            temp = solution[swap_pos_1]
            solution[swap_pos_1] = solution[swap_pos_2]
            solution[swap_pos_2] = temp