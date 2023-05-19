import numpy as np
from GeneticAlgo import GeneticAlgo
from DarwinAlgo import DarwinAlgo
from LamarckAlgo import LamarckAlgo
import time
import multiprocessing as mp

def pickle_eval_word(args):
    word, word_set, letter_freq, pair_freq = args[:4]
    word_coeff, letter_coeff, pairs_coeff = args[4:]
    valid_word = 0
    letter_score = 0
    pair_score = 0
    length = len(word)
    if word in word_set:
            # criteriea 1: words in dict/words in message
            valid_word = length
    for i in range(len(word)):
        # criteriea 2: sum of frequencies of single letters
        letter_score += letter_freq[word[i]]
        if i+1 < len(word):
            # criteriea 3: sum of frequencies of pairs of letters
            pair_score += pair_freq[word[i:i+2]]
    
    letter_score = letter_score / length
    pair_score = pair_score / length
    score = (word_coeff*valid_word +
            letter_coeff*letter_score + pairs_coeff*pair_score)
    return score
    

if __name__ == "__main__":
    
    alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])
    sol_rep = np.arange(26)
    letter_freq = dict()
    pair_freq = dict()

    with open("enc.txt") as encrypted_file:
        enc_mess = encrypted_file.read()

    with open("dict.txt") as word_dict:
        # words = set(word_dict.readlines())
        text = word_dict.read()
        words = set(text.split("\n"))

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
    replication_rate = 0.1
    cross_over_rate = 1-replication_rate
    mutation_rate = 0.04
    word_coeff = 20
    letter_coeff = 5
    pair_coeff = 13
    
    mp.set_start_method('spawn')
    # params = [[20, 5, 13], [20, 13, 5],[10, 7, 3],[10, 3, 7],
    #          [10, 3, 1], [5, 3, 1],[5, 1, 0],[3, 1, 0],[1, 0, 0]]
    # params = [75, 100, 125, 150, 200, 250, 300]
    params = [150]
    with mp.Pool(60) as executor:
        for param in params:
            # word_coeff, letter_coeff ,pair_coeff = param
            gen_size = param
            algo_settings = [enc_mess, letter_freq, pair_freq, words,
                            replication_rate, cross_over_rate,
                            mutation_rate, gen_size, executor,
                            pickle_eval_word, word_coeff, 
                            letter_coeff, pair_coeff, 5]
                

            genetic_algo = DarwinAlgo(*algo_settings)
            
            start = time.time()
            solution, fitness_count = genetic_algo.run()
            
            end = time.time()
            plain_text = genetic_algo.decode_message(enc_mess, solution)
            score = genetic_algo.eval_func(solution)
            cover = genetic_algo.coverage(solution)
            with open("param_results.csv", 'a') as res:
                # res.write(f"{word_coeff},{letter_coeff},{pair_coeff},{fitness_count}\n")
                res.write(f"{gen_size},{fitness_count},{score},{cover}")
        
    with open("perm.txt", 'w+') as gen_perm:
        for i in range(len(solution)):
            letter = alphabet[solution[i]]
            gen_perm.write(f"{alphabet[i]} {letter}\n")
    with open("plain.txt", 'w+') as gen_sol:
        gen_sol.write(plain_text)
        
    print(end-start)
   
