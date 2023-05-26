import numpy as np
from GeneticAlgo import GeneticAlgo
from DarwinAlgo import DarwinAlgo
from LamarckAlgo import LamarckAlgo
import matplotlib.pyplot as plt


def encode_message(message, alphabet):
    rng = np.random.default_rng(7)
    permutaion = rng.permutation(alphabet)
    table = str.maketrans("".join(alphabet), "".join(permutaion))
    new_message = message.translate(table)
    return new_message


def graph_stats(stats, args):
    avg_score_data = stats['avg']
    max_score_data = stats['max']
    iterations = np.arange(stats.size)
    plt.plot(iterations, avg_score_data, color='g', label="average score")
    plt.plot(iterations, max_score_data, color='r', label="max score")
    plt.legend()
    name_string = ",".join(str(args))
    plt.savefig(f"{name_string}.png")
    plt.clf()

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

    gen_size = 100
    replication_rate = 0.1
    cross_over_rate = 1-replication_rate
    mutation_rate = 0.8
    mutation_number = 3
    word_coeff = 6
    letter_coeff = 2
    pair_coeff = 1

    # enc_mess = encode_message(enc_mess, alphabet)
    # params = [[15, 3, 2], [5, 3, 1],[2, 1, 1],[2, 1, 1],[1, 1, 1],[1, 0, 0]]
    # params = [75, 100, 125, 150, 200, 250, 300]

    with open("param_results.csv", 'a') as res:
        res.write("word_coeff,letter_coeff,pair_coeff,fitness_count,score,cover:\n")
    
    # for param in params:
        # word_coeff, letter_coeff ,pair_coeff = param
        # gen_size = param
        algo_settings = [enc_mess, letter_freq, pair_freq, words,
                        replication_rate, cross_over_rate,
                        mutation_rate, mutation_number, gen_size, word_coeff, 
                        letter_coeff, pair_coeff]
            

        genetic_algo = GeneticAlgo(*algo_settings)
        solution, fitness_count, stats = genetic_algo.run(500)
    
        if stats.size == 500:
            with open("param_results.csv", 'a') as res:
                res.write(f"{word_coeff},{letter_coeff},{pair_coeff},over 500!\n")
            # continue
        graph_stats(stats, [word_coeff, letter_coeff ,pair_coeff])

        plain_text = genetic_algo.decode_message(enc_mess, solution)
        score = genetic_algo.eval_func(solution)
        cover = genetic_algo.coverage(solution)

        with open("param_results.csv", 'a') as res:
            res.write(f"{word_coeff},{letter_coeff},{pair_coeff},{fitness_count},{score},{cover}\n")
            # res.write(f"{gen_size},{fitness_count},{score},{cover}\n")
    
    with open("perm.txt", 'w+') as gen_perm:
        for i in range(len(solution)):
            letter = alphabet[solution[i]]
            gen_perm.write(f"{alphabet[i]} {letter}\n")

    with open("plain.txt", 'w+') as gen_sol:
        gen_sol.write(plain_text)
        
   
