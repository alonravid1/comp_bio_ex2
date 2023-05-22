# Genetic Decryption Algorithm
*author* Alon Ravid

## Table of Contents
1. [Solution Representation and Evaluation](#sol)
2. [Evolution of Solutions](#evolve)
3. [Early Convergance and Halting](#conv)
4. [Algorithm Analysis](#algo)

In this report I will detail my implementation of a genetic algorithm which decrypts a message encrypted via alphabet permution. The underlying assumptions are that files containing a dicionary of frequent words in english, a letter freqency and letter pairs frequency files are given. In addition upper/lower case do not matter, therefore all input is converted to lower case.

In addition to these files, the algorithm will be tested on a range of parameters:
* Generation Size
* Replication Rate
* Crossover Rate
* Mutation Rate
* Score Variables Coefficients
* Algorithm Type


## Solution Representation and Evaluation <a name=sol></a>
In this exercise, an encrypted message is created via a permutation of the alphabet, for example swapping 
all instances of 'a' with 'c'. To break the encryption is to have the permutation table which was used for the encryption, and applying it backwards, meaning a dictionary of characters as keys and also as values is a valid representation to the solutions of this probelm.

At first I attempted to represent the solutions this way, but after several runs I had decided to optimize the running time by replacing the dictionaries and direct representation by characters, to a representation of solutions using a numpy array of integers between 0 and 25 which are then shifted around. If in the first place of the array is the number 2, that means the algorithm will swap 'a' with 'c', which is the letter of the 2nd cell in the array containing the alphabet.

To evaluate a solution, I applied the permutation as described above on the encoded message, and then I calculated a solution's score by evaluating each word and summing their scores. A word's score was calculated based on whether or not it was a valid word from the given dictionaries, the sum of its letter frequencies and the sum of pairs of letters frequencies, each multiplyed by a parameter coefficient as described in the following equation:

$score = a \cdot 1_{is\_valid\_word} \cdot word\_length + b \cdot \text{letters\_freq} + c \cdot \text{pairs\_freq}$

## Evolution of Solutions <a name=evolve></a>
In order to apply evolutionary pressure over a generation of solutions while creating a greater variance of solutions, 3 rules were applied during the creation of a new generation:

1.The best solutions was copied "as is".
2.The rest of the solutions were created by crossing over two solutions at a time. The solutions were picked via linear sampling, that is the chance to pick a solution was equal to its score's proportion out of the sum of all scores.
3.All solutions generated via crossovers were mutated, where each letter had a small chance to be randomly swapped with another letter in the solution array.

The crossing over was applied by taking two arrays, randomly generating a number in between 1 and 25, and then "cutting" both arrays at that index, and then gluing them together such that the first solution's first part is glued to the second solution's second part, and the second solution's first part is glued to the first solution's second part.

Afterwards, the solutions's validty was checked and fixed by searching for double letter and missing letters that can be created this way, swapping the first instace of a double letter with a missing letter.

## Early Convergance Problem and Halting<a name=conv></a>
A problem facing genetic algorithms is an early convergence to a local maximum point. Because the solution with the highest score is replicated multiple times at the next generation, and a solution's chance to be chosen for crossover is defined as the porportion of its score to the total score of all solutions, the highest scoring ones can replace the majority of other solutions.

When this happens, we remain with a much less diverse population which is concentrated around a local maximum, without the abilty to overcome it and thus becoming stuck there. In order to overcome this problem, I programmed the algorithm to follow the following logic:

Check wether the best solution had changed in the last 5 generation. If not, calculate the coverage of valid words over the decoded message words:
* If the coverage is lower than 50%, that means the best solution is stuck but the current solution is far from the optimal one, since a large majority of  the message's words should be found in the dictionary. In this case the algorithm "resets", generating a new random generation and continues from there to the next generation, without reseting the fitness count.
* If the coverage is lower than 90% but above 50%, that means in most likelihood that the current solution is good, but has a few words with less frequent letters which are still wrong. To correct this slight difference, the letter pairs frequency's coefficient is increased, and the single letter's is decreased, with mutation rate increasing as well to diversify the solution space in hope of getting the last few letter swaps required.

The algorithm stops running and returns the current best solution once it has not changed for the last 10 generations, and the coverage is higher than 90%.

## Algorithm Analysis<a name=algo></a>
At first I 