# Genetic Algorithms
These genetic algorithm for decoding a message encrypted by permutating the alphabet, were written as an exercise in computational biology, in which I wrote and tuned a generic genetic algorithm,
a Darwin variant and a Lamarck variant. The program can run simply by clicking the executable file, upon which a GUI in which the different parameters can be set, an algorithm type selected and
when running it it will decode the message in the enc.txt file, using the dict.txt, Letter_Freq.txt and Letter2_freq.txt which needs to be in the same directory as the executable file.


All three algorithms are detailed in the report, as well as the parameters which I found to be optimal for running them on the encoded message, as well as on several other test. Upon finishing decoding a message,
the program will create two files named perm.txt and plain.txt, in which the permutation used to encode the message and the decoded message will be written respectively.

IMPORTANT NOTE: Once an algorithm is started, the GUI might faulter and not update every iteration, or appear to be stuck and the program not responding, THE ALGORITHM IS STILL RUNNING and once it finishes it will generate the files as written above
and open a popup with the iterations log and best solution statistics.
