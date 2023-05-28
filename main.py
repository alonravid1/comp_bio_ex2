import numpy as np
from GeneticAlgo import GeneticAlgo
from DarwinAlgo import DarwinAlgo
from LamarckAlgo import LamarckAlgo
import matplotlib.pyplot as plt
import multiprocessing as mp

def pickle_eval_word(args):
    """the function evaluates a single word's score according to
    wether or not it appears in the given dictionary, its length and
    letter and pairs of letters' frequency.
    
    it is sent to the algorithms through main due to the fact that
    the solution evaluation function uses multiprocessing to go over multiple
    words at once and greatly reduce runtime, but for a function to be used
    in multiprocessing it must be pickleable, which can be achived by defining
    it in the first indent level.

    Args:
        word (string): a word to be evaluated
        word_dict (set): a dictionary of valid words indexed by lengths

    Returns:
        score: a number representing a word's score
    """


    word, word_dict = args[:2]
    min_hamming_distance = 1

    length = len(word)
    if word in word_dict[length]:
            return 1

    for dict_word in word_dict[length]:
        hamming_distance = sum(char1 != char2 for char1, char2 in zip(word, dict_word)) / length
        min_hamming_distance = min(min_hamming_distance, hamming_distance)

    return (1 - min_hamming_distance)
    
    
def graph_stats(stats, param):
    avg_score_data = stats['avg']
    max_score_data = stats['max']
    iterations = np.arange(stats.size)
    plt.plot(iterations, avg_score_data, color='g', label="average score")
    plt.plot(iterations, max_score_data, color='r', label="max score")
    plt.legend()
    plt.savefig(f"{param[0]}, {param[1]}, {param[2]}.png")
    plt.clf()

if __name__ == "__main__":
    
    alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])
    sol_rep = np.arange(26)
    letter_freq = dict()
    pair_freq = dict()

    with open("enc.txt") as encrypted_file:
        enc_mess = encrypted_file.read()

    with open("dict.txt") as word_dict:
        text = word_dict.read()
        words = text.split("\n")
        word_dict = dict({i:[] for i in range(1,45)})
        for word in words:
            word_dict[len(word)].append(word)
        for key in word_dict.keys():
            word_dict[key] = set(word_dict[key])
            
            

    with open("Letter_Freq.txt") as letter_freq_file:
        for line in letter_freq_file.readlines():
            line = line.strip("\n")
            freq, letter = line.split("\t")
            letter_freq[letter.lower()] =  float(freq)

    with open("Letter2_Freq.txt") as pair_freq_file:
        for line in pair_freq_file.readlines():
            line = line.strip("\n")
            if line == "\t" or line == "\t#REF!":
                break
            freq, pair = line.split("\t")
            pair_freq[pair.lower()] = float(freq)

    gen_size = 150
    replication_rate = 0.25
    cross_over_rate = 1-replication_rate
    mutation_rate = 0.8
    mutation_number = 3
    word_coeff = 20
    letter_coeff = 5
    pair_coeff = 13
    
    repeats = 1
    mp.set_start_method('spawn')
    params = [[30,1,1]]
    # params = [[10,7,3],[5,1,0],[3,1,0],[5,3,1],[1,1,1]]
    # params = [75, 100, 125, 150, 200, 250, 300]
    with open("param_results.csv", 'a') as res:
        res.write("word_coeff,letter_coeff,pair_coeff,fitness_count,score,cover:\n")
        
    with mp.Pool(60) as executor:
        for param in params:
            avg_score = 0
            avg_fitness = 0
            avg_cover = 0
            word_coeff, letter_coeff ,pair_coeff = param
            # gen_size = param

            algo_settings = [enc_mess, letter_freq, pair_freq, word_dict,
                                replication_rate, cross_over_rate,
                                mutation_rate, mutation_number, gen_size, executor,
                                pickle_eval_word, word_coeff, 
                                letter_coeff, pair_coeff]
            
            for i in range(repeats):
                genetic_algo = GeneticAlgo(*algo_settings)
                # sol1 = np.random.permutation(26)
                # sol2 = np.random.permutation(26)
                # genetic_algo.cross_over(sol1, sol2)
                solution, fitness_count, stats = genetic_algo.run(360)
                if i == 0:
                    graph_stats(stats, param)
                    plain_text = genetic_algo.decode_message(enc_mess, solution)
                
                avg_score += genetic_algo.eval_func(solution)
                avg_cover += genetic_algo.coverage(solution)
                avg_fitness += fitness_count

            avg_score = round(avg_score / repeats, 2)
            avg_cover = round(avg_cover / repeats, 2)*100
            avg_fitness = avg_fitness // repeats
            with open("param_results.csv", 'a') as res:
                res.write(f"{word_coeff},{letter_coeff},{pair_coeff},{avg_fitness},{avg_score},{avg_cover}%\n")
                # res.write(f"{gen_size},{fitness_count},{score},{cover}\n")
        
    with open("perm.txt", 'w+') as gen_perm:
        for i in range(len(solution)):
            letter = alphabet[solution[i]]
            gen_perm.write(f"{alphabet[i]} {letter}\n")
    with open("plain.txt", 'w+') as gen_sol:
        gen_sol.write(plain_text)
        
   
