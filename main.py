import numpy as np
from Algo import Algo
import time
import multiprocessing as mp


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

    gen_size = 400
    replication_rate = 0.05
    cross_over_rate = 1-replication_rate
    mutation_rate = 0.05
    word_coeff = 10
    letter_coeff = 4
    pair_coeff = 1
    
    mp.set_start_method('spawn')
    with mp.Pool(60) as executor:
        algo_settings = [enc_mess, letter_freq, pair_freq, words,
                        replication_rate, cross_over_rate,
                        mutation_rate, gen_size, executor,
                        word_coeff, letter_coeff, pair_coeff]
            

        genetic_algo = Algo(*algo_settings)
        
        start = time.time()
        solutions = genetic_algo.run(450)
        
        end = time.time()
        plain_text = genetic_algo.decode_message(enc_mess, solutions[-1])
        print(plain_text)
        print(alphabet[solutions[-1]])
        
        
    with open("perm.txt", 'w+') as gen_perm:
        for i in range(len(solutions[-1])):
            letter = alphabet[solutions[-1][i]]
            gen_perm.write(f"{alphabet[i]} {letter}\n")
    with open("plain.txt", 'w+') as gen_sol:
        gen_sol.write(plain_text)
        
    print(end-start)
   
