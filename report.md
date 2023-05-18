# Genetic Decryption Algorithm
*author* Alon Ravid

## Table of Contents
1. [Solution Representation](#sol)
2. [Evolution of Solutions](#evolve)
3. [Early Convergance Problem](#conv)
4. [Algorithm Analysis](#algo)

In this report I will detail my implementation of a genetic algorithm which decrypts a message encrypted via alphabet permution. The underlying assumptions are that files containing a dicionary of frequent words in english, a letter freqency and letter pairs frequency files are given. In addition upper/lower case do not matter, therefore all input is converted to lower case.

In addition to these files, the algorithm will be tested on a range of parameters:
* Replication Rate
* Cross Over Policy
* Mutation Rate

## Solution Representation and Evaluation <a name=sol></a>
In this exercise, an encrypted message is created via a permutation of the alphabet, for example swapping 
all instances of 'a' with 'c'. To break the encryption is to have the permutation table which was used for the encryption, and applying it backwards, meaning a dictionary of characters as keys and also as values is a valid representation to the solutions of this probelm.

At first I attempted to represent the solutions this way, but after several runs I had decided to optimize the running time by replacing the dictionaries and direct representation by characters, to a representation of solutions using a numpy array of integers between 0 and 25 which are then shifted around. If in the first place of the array is the number 2, that means the algorithm will swap 'a' with 'c', which is the letter of the 2nd cell in the array containing the alphabet.

To evaluate a solution, I applied the permutation as described above on the encoded message, and then I calculated a score based on the number of valid words from the given dictionaries, plus the sum of letter frequencies and the sum of pairs of letters frequencies, summed in the equation:

$score = a \cdot \text{words / word length} + b \cdot \text{letters\_freq} + c \cdot \text{pairs\_freq}$


I played around with the coefficients, and at

## Evolution of Solutions <a name=evolve></a>
In order to apply evolutionary pressure over a generation of solutions while creating a greater variance of solutions, 3 rules were applied during the creation of a new generation:

1.A few of the best solutions were copied "as is"
2.The rest of the solutions were created by crossing over two solutions at a time. The solutions were picked by linear sampling, that is the chance to pick a solution was equal to its score's proportion out of the sum of all scores.
3.All solutions generated via cross over were mutated, where each letter had a small chance to be randomly swapped with another letter in the solution array.

The crossing over was applied by taking two arrays, randomly generating a number in between 1 and 25, and then "cutting" both arrays at that index, and then gluing them together such that the first solution's first part is glued to the second solution's second part, and the second solution's first part is glued to the first solution's second part.

Afterwards, the solutions's validty was checked and fixed by searching for double letter and missing letters that can be created this way, swapping the first instace of a double letter with a missing letter.

## Early Convergance Problem <a name=conv></a>
A problem facing a genetic algorithm is a similar one to an early convergence to a local maximum. Because a solution with the highest score is replicated and a solution's chance to be chosen for crossover is defined as the porportion of its score to the total score of all solutions, the highest scoring ones can replace the majority of other solutions.

When this happens, we remain with a much less diverse population which is concentrated around a local maximum, without the abilty to overcome it and becoming stuck there. In order to overcome this problem,
