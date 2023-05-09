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
The encryption of the prolem the algorithm sets to break is created via a permutation of the alphabet, meaning it can be encoded in a dictionary where single alphabet characters are the keys and values.