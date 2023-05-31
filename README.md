# Genetic Algorithms
These genetic algorithm for decoding a message encrypted by permutating the alphabet, were written as an exercise in computational biology, in which I wrote and tuned a generic genetic algorithm,
a Darwin variant and a Lamarck variant.

In order to run the program, you must make sure the following files are in the same directory:
* Run Genetic Algorithm.exe
* enc.txt - containsthe encoded message
* dict.txt - contains a dictionary of valid words
* Letter_Freq.txt - contains a dictionary of letters and their frequency in the English language
* Letter2_Freq.txt - contains a dictionary of pairs of letters and their frequency in the English language

Once you have those, simply click the executable file, upon which a GUI in which the different parameters can be set, an algorithm type selected and started. The "Information" button will give more detail about them.

All three algorithms are detailed in the report, as well as the parameters which I found to be optimal for running them on the encoded message, and on several other test encoded messages. Upon finishing decoding a message, the program will create two files named perm.txt and plain.txt, in which the permutation used to encode the message and the decoded message will be written respectively.

IMPORTANT NOTE: Once an algorithm is started, the GUI might faulter and not update every iteration, or appear to be stuck and the program not responding, THE ALGORITHM IS STILL RUNNING and once it finishes it will generate the files as written above
and open a popup with the iterations log and best solution statistics.
