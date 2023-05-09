import numpy as np
import re

def validate_solution(perm_dict):
    """validate and fix a permutaion dictionary after a crossover,
    replacing one instance of a letter appearing twice with a missing letter

    Args:
        perm_dict (dict): permutaion dictionary
    """

    vals = list(perm_dict.values())
    double_letters = []
    missing_letters = []

    for char in alphabet:
        if vals.count(char) == 2:
            double_letters.append(char)

        if vals.count(char) == 0:
            missing_letters.append(char)

    for char in double_letters:
        for key in perm_dict.keys():
            if perm_dict[key] == char:
                perm_dict[key] = missing_letters.pop()
                break

    


def cross_over(perm_dict1, perm_dict2):
    """swaps two dictionaries at a random point,
    then fixing any double or missing alphabet letters
    from the resulting solutions

    Args:
        perm_dict1 (dict): permutation dictionary

        perm_dict2 (dict): permutation dictionary
    """

    # choose random point in the dict
    # to swap the dictionaries, with at least 1 swap
    crossing_point = np.random.randint(1,25)
    vals1 = list(perm_dict1.values())
    vals2 = list(perm_dict2.values())


    perm_dict1 = dict(zip(alphabet[:crossing_point], vals1[:crossing_point]))
    perm_dict1.update(dict(zip(alphabet[crossing_point:], vals2[crossing_point:])))

    perm_dict2 = dict(zip(alphabet[:crossing_point], vals2[:crossing_point]))
    perm_dict2.update(dict(zip(alphabet[crossing_point:], vals1[crossing_point:])))

    validate_solution(perm_dict1)
    validate_solution(perm_dict2)


def apply_perm(message, perm_dict):
    """apply permutation to message according to the
    permutation dictionary

    Args:
        message (string): message to be permutated
        perm_dict (dict): dictionary of permutations

    Returns:
        string: message after applying permutation dictionary
    """

    new_message = ""
    for char in message:
        # check if character is in dictionary, i.e. it is not a special character
        new_char = perm_dict.get(char)
        if new_char != None:
            new_message += new_char
        else:
            new_message += char
            
    return new_message

def eval(message, solution):
    decrypt_message = apply_perm(message, solution)


    # for evaluation, remove all non abc characters
    decrypt_message =  re.sub('[0-9\[\](){}<>;@&^%$!*?,.\n]', '', decrypt_message)
    message_words = decrypt_message.split(" ")

    # remove any empty words
    while message_words.count("") != 0:
        message_words.remove("")

    valid_word_count = 0
    letter_score = 0
    pair_score = 0

    for word in message_words:
        if word in words:
            # criteriea 1: words in dict/words in message
            valid_word_count += 1
        for i in range(len(word)):
            # criteriea 2: sum of frequencies of single letters
            letter_score += letter_freq[word[i]]
            if i+1 < len(word):
                # criteriea 3: sum of frequencies of pairs of letters
                pair_score += pair_freq[word[i:i+2]]
    
    valid_word_count = valid_word_count / len(message_words)
    letter_score = letter_score / len(decrypt_message)
    pair_score = pair_score / len(decrypt_message)
    print(valid_word_count)
    print(letter_score)
    print(pair_score)

    
if __name__ == "__main__":
    encrypted_file = open("enc.txt")
    enc_mess = encrypted_file.read()

    alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    letter_freq = dict()
    pair_freq = dict()

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

    perm1 = np.random.permutation(alphabet)
    perm2 = np.random.permutation(alphabet)

    dict1 = dict(zip(alphabet, perm1))
    dict2 = dict(zip(alphabet, perm2))

    print(apply_perm(enc_mess, dict1))
    # mess = apply_perm(enc_mess, dict1)
    # cross_over(dict1, dict2)
    eval(enc_mess, dict1)