import numpy as np
from GeneticAlgo import GeneticAlgo
from DarwinAlgo import DarwinAlgo
from LamarckAlgo import LamarckAlgo
import matplotlib.pyplot as plt
import multiprocessing as mp
from Gui import Gui

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
    word, word_set = args[:2]
    length = len(word)
    
    if word in word_set:
            return 1, word
    
    min_hamming_distance = 1
    for set_word in word_set:
        hamming_distance = sum(char1 != char2 for char1, char2 in zip(word, set_word))/length
        min_hamming_distance = min(min_hamming_distance, hamming_distance)
    
    if min_hamming_distance > 0.5:
        return 0, word
    
    return (1 - min_hamming_distance), word
    
if __name__ == "__main__":
    mp.freeze_support()
    gui = Gui(pickle_eval_word)
    gui.start()