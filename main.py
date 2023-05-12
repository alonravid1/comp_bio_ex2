import numpy as np
from Algo import Algo

if __name__ == "__main__":
    
    alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])
    sol_rep = np.arange(26)
    letter_freq = dict()
    pair_freq = dict()

    with open("enc.txt") as encrypted_file:
        enc_mess = encrypted_file.read()

    with open("dict.txt") as word_dict:
        words = set(word_dict.readlines())

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
    mutation_rate = 0.1

    algo_settings = [enc_mess, letter_freq, pair_freq, words,
                      replication_rate, cross_over_rate, mutation_rate, gen_size]
        

    genetic_algo = Algo(*algo_settings)
    solutions = genetic_algo.run(10)

    print(genetic_algo.decode_message(enc_mess, solutions[0]))
    print(alphabet[solutions[0]])

    # rng = np.random.default_rng(7)
    # solutions = np.tile(sol_rep, (10, 1))
    # for i in solutions:
    #     # this is done in a for loop because shuffling all
    #     # at once shuffles them the same way
    #     rng.shuffle(i)
    # new_solutions = np.zeros((10, 26))
    # print(solutions.shape)
    # print(new_solutions.shape)



    # print(apply_perm(enc_mess, dict1))
    # mess = apply_perm(enc_mess, dict1)
    # cross_over(dict1, dict2)
