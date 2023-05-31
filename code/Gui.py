import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
from GeneticAlgo import GeneticAlgo
from DarwinAlgo import DarwinAlgo
from LamarckAlgo import LamarckAlgo
import time
import os
import multiprocessing as mp   

class Gui:
    def __init__(self, pickle_eval_word):
        """initialise the GUI, start the main loop and show main screen

        Args:
            algo (class, optional): type of genetic algorithm to run
        """        
        
        # setting general GUI properties
        sg.theme('DarkAmber')
        
        self.AppFont = 'Any 14' # font
        self.pickle_eval_word = pickle_eval_word
        self.algorithm = "Genetic Algorithm"
        self.show_graph_flag = False

        # set main window's layout
        self.main_layout = [
            [sg.Button('Information', font=self.AppFont)],
            # set parameters
            [sg.Text('Enter parameters:', font='Any 18')],
            [sg.Text('Generation Size:',font=self.AppFont), sg.Input(key='gen_size', size=(15,1), font=self.AppFont, default_text='100')],
            [sg.Text('Replication Rate:',font=self.AppFont), sg.Input(key='replication_rate', size=(15,1), font=self.AppFont, default_text='0.3'),
             sg.Text('Mutation Rate:',font=self.AppFont), sg.Input(key='mutation_rate', size=(15,1), font=self.AppFont, default_text='0.8'),
             sg.Text('Number of Mutations:',font=self.AppFont), sg.Input(key='mutation_number', size=(15,1), font=self.AppFont, default_text='5'),
             sg.Text('Number of Swaps:',font=self.AppFont), sg.Input(key='swaps_number', size=(15,1), font=self.AppFont, default_text='3')
             ],
             
      
            [sg.Text('Word Weight',font=self.AppFont), sg.Input(key='word_weight', size=(15,1), font=self.AppFont, default_text='3'),
             sg.Text('Letter Weight',font=self.AppFont), sg.Input(key='letter_weight', size=(15,1), font=self.AppFont, default_text='1'),
             sg.Text('Pairs Weight',font=self.AppFont), sg.Input(key='pairs_weight', size=(15,1), font=self.AppFont, default_text='0')
             ],
            
            # set algorithm type
            [sg.Text('Algorithm Type', font=self.AppFont),
              sg.Button('Genetic Algorithm', font=self.AppFont, disabled=True),
              sg.Button('Darwin Variation', font=self.AppFont),
              sg.Button('Lamarck Variation', font=self.AppFont)
            ],

            [sg.Button('Show Graph', font=self.AppFont),
             sg.Button('Dont Show Graph', font=self.AppFont, disabled=True)],

            # run different kinds of simulation
            [sg.Button('Start Decoding', font=self.AppFont)],
            [sg.Button('Generate Stats', font=self.AppFont)],

            [sg.Button('Exit', font=self.AppFont)]
            ]
        
        self.infotext = """ The program will start decoding a message once you press 'Start Decoding'.
Once preseed, a new window will pop up in which it will attempt to show each generation's statistics and the start of the message according to the best solution.

IMPORTANT: The window might not update each iteration, or will appear to be stuck or not respond. The algorithm is still running in the background, and will open a window with the log of all iterations and of the best solution and statistics once it finishes.

Please enter the following parameters as follows:
* Generation Size (int)
* Replication Rate (float between 0 and 1)
* Mutation Rate (float between 0 and 1)
* Number of Mutations (int)
* Score Weights (float)

Show Graph will show at the end of a run a graph of the maximum and average score per iteration, and a graph of the coverage of the best solution per iterations
* Generate Stats will run the algorithm 10 times and then present the average values of those runs.
"""
    
    def create_run_layout(self):
        """
        generates a simulation screen layout.
        the gui package cannot reuse a layout, generating a new
        layout with new element objects is the solution, rather
        than trying to create a template and starting to work on
        deep copying the very flexible nested arrays of elements.
        """
        fresh_layout = [
                    [sg.Text("Please wait, starting running", key='header')],
                    [sg.Text("", key='decoded_mess')],
                    # [sg.MLine(size=(90,30), key='-ML-'+sg.WRITE_ONLY_KEY, disabled=True)],
                    [sg.Button('Exit')]
                    ]
        
        return fresh_layout
        
    def graph_stats(self, stats):
        avg_score_data = stats['avg']
        max_score_data = stats['max']
        max_cover_data = stats['cover']

        iterations = np.arange(stats.size)
        fig, axs = plt.subplots(2,1)
        fig.set_layout_engine('tight')
        axs[0].plot(iterations, avg_score_data, color='g', label="average score")
        axs[0].plot(iterations, max_score_data, color='r', label="max score")
        axs[0].set(xlabel="Iteration Number", ylabel="Score")
        axs[0].legend()

        axs[1].plot(iterations, max_cover_data, color='b', label="max coverage")
        axs[1].set(xlabel="Iteration Number", ylabel="Coverage")
        axs[1].legend()

        fig.savefig("graph.png")
        fig.clf()

    
    def get_files(self):
        letter_freq = dict()
        pair_freq = dict()

        with open("enc.txt") as encrypted_file:
            enc_mess = encrypted_file.read()
            enc_mess = enc_mess.lower()

        with open("dict.txt") as word_dict:
            text = word_dict.read()
            words = text.split("\n")
            word_dict = dict({i:[] for i in range(1,45)})
            for word in words:
                word_dict[len(word)].append(word)
            for key in word_dict.keys():
                word_dict[key] = set(word_dict[key])

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

        return enc_mess, word_dict, letter_freq, pair_freq
    
    def generate_stats(self, algo_values):
        """
        creates the new simulation window, runs the simulation
        to get all itertion frames, and then show each one
        using draw_frame(), afterwhich it gives the user the option to
        close the window whenever they'd like

        Args:
            algo_values (nparray): array of the simulation parameters
        """        
        enc_mess, word_dict, letter_freq, pair_freq = self.get_files()

        # copy while all items are numbers
        new_algo_values = algo_values.copy()

        if enc_mess == "" or len(word_dict) == 0 or len(letter_freq) == 0 or len(pair_freq) == 0:
            sg.popup("Error parsing files!", self.AppFont)
            return
        
        new_algo_values += [enc_mess, word_dict, letter_freq, pair_freq]

        window = sg.Window('Evolution Presentation',
                                    self.create_run_layout(),
                                    finalize=True,
                                    resizable=True,
                                    element_justification="left")
        
        with mp.Pool() as executor:
            new_algo_values += [executor, self.pickle_eval_word]
            if self.algorithm == 'Genetic Algorithm':
                algo = GeneticAlgo(*new_algo_values)

            if self.algorithm == 'Darwin Variation':
                algo = DarwinAlgo(*new_algo_values)

            if self.algorithm == 'Lamarck Variation':
                algo = LamarckAlgo(*new_algo_values)
            
            avg_score = 0
            avg_cover = 0
            avg_fitness = 0

            repeats = 5
            for i in range(repeats):
                finished_flag = False
                algo.init_run(150)
                while not finished_flag:
                    event, values = window.read(timeout=2)

                    if event == sg.WIN_CLOSED or event == 'Exit':
                        window.close()
                        return

                    returned_stats = algo.iterate_step()
                    solution, fitness_count, stats = returned_stats[:3]
                    iteration, finished_flag = returned_stats[3:]

                avg_score += stats[-1]['max']
                avg_cover += stats[-1]['cover']
                avg_fitness += fitness_count
                    
                   
            window['header'].update("Done!")
            avg_score = round(avg_score / repeats, 2)
            avg_cover = round(avg_cover / repeats, 2)
            avg_fitness = avg_fitness // repeats
            
            sg.popup(f"{avg_fitness},{avg_score},{avg_cover}%\n",title="Results")          

            alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])
            plain_text = algo.decode_message(solution, enc_mess)
            with open("perm.txt", 'w+') as gen_perm:
                for i in range(len(solution)):
                    letter = alphabet[solution[i]]
                    gen_perm.write(f"{alphabet[i]} {letter}\n")

            with open("plain.txt", 'w+') as gen_sol:
                gen_sol.write(plain_text)

            while True:
                event, values = window.read(timeout=2)

                if event == sg.WIN_CLOSED or event == 'Exit':
                    window.close()
                    return
                
    def run_algorithm(self, algo_values):
        """
        creates the new simulation window, runs the simulation
        to get all itertion frames, and then show each one
        using draw_frame(), afterwhich it gives the user the option to
        close the window whenever they'd like

        Args:
            algo_values (nparray): array of the simulation parameters
        """        
        enc_mess, word_dict, letter_freq, pair_freq = self.get_files()

        # copy while all items are numbers
        new_algo_values = algo_values.copy()

        if enc_mess == "" or len(word_dict) == 0 or len(letter_freq) == 0 or len(pair_freq) == 0:
            sg.popup("Error parsing files!", self.AppFont)
            return
        
        new_algo_values += [enc_mess, word_dict, letter_freq, pair_freq]

        window = sg.Window('Evolution Presentation',
                                    self.create_run_layout(),
                                    finalize=True,
                                    resizable=True,
                                    element_justification="left")
        
        with mp.Pool() as executor:
            new_algo_values += [executor, self.pickle_eval_word]
            if self.algorithm == 'Genetic Algorithm':
                algo = GeneticAlgo(*new_algo_values)

            if self.algorithm == 'Darwin Variation':
                algo = DarwinAlgo(*new_algo_values)

            if self.algorithm == 'Lamarck Variation':
                algo = LamarckAlgo(*new_algo_values)

            finished_flag = False
            algo.init_run(150)
            final_output = ""

            while not finished_flag:
                event, values = window.read(timeout=2)

                if event == sg.WIN_CLOSED or event == 'Exit':
                    window.close()
                    return

                returned_stats = algo.iterate_step()
                solution, fitness_count, stats = returned_stats[:3]
                iteration, finished_flag = returned_stats[3:]

                plain_text = algo.decode_message(solution, enc_mess)
                decoded_mess = plain_text[:101]

                if fitness_count == -1:
                    final_output += f"reset at iteration {iteration}\n"
                    window['header'].update(f"reset at iteration {iteration}\n")
                    window['decoded_mess'].update("")
                elif not finished_flag:
                    final_output += f"generation {iteration}, max score {stats[0]['max']}, max coverage {stats[0]['cover']}%\n"
                    final_output += decoded_mess + "\n\n"
                    window['header'].update(f"generation {iteration}, max score {stats[0]['max']}, max coverage {stats[0]['cover']}%\n")
                    window['decoded_mess'].update(decoded_mess)

                
                
                
            final_output += f"best solution found at generation {iteration}, in {fitness_count} evaluations\n"
            final_output += f"max score: {stats[-1]['max']}, max coverage: {stats[-1]['cover']}%\n"
            final_output += decoded_mess + "\n"
            window['header'].update("Done!")
            window['decoded_mess'].update(f"finished at iteration {iteration}")
            if self.show_graph_flag:  
                self.graph_stats(stats)
                sg.popup_scrolled(final_output, image="graph.png",title="Iterations Log")
            else:
                sg.popup_scrolled(final_output,title="Iterations Log")


            alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])

            with open("perm.txt", 'w+') as gen_perm:
                for i in range(len(solution)):
                    letter = alphabet[solution[i]]
                    gen_perm.write(f"{alphabet[i]} {letter}\n")

            with open("plain.txt", 'w+') as gen_sol:
                gen_sol.write(plain_text)

            while True:
                event, values = window.read(timeout=2)

                if event == sg.WIN_CLOSED or event == 'Exit':
                    window.close()
                    return


        
    def process_values(self, values):
        """_summary_

        Args:
            values (nparray): array of simulation parameters as entered by the user

        Returns:
            nparray: an array of rounded values, confirming to the parameters conditions
        """        
        try:
            algo_values = [float(values['replication_rate']), (1 -  float(values['replication_rate'])),
                        float(values['mutation_rate']), int(values['mutation_number']), int(values['gen_size']),
                        float(values['word_weight']), float(values['letter_weight']),
                        float(values['pairs_weight']), int(values['swaps_number'])]
            
            # check all values are positive
            if any(value < 0 for value in algo_values):
                sg.popup('Values cannot be negative')
                return None
            if (algo_values[3] < 1 or algo_values[4] < 1 or algo_values[-1] < 1):
                sg.popup('mutation number, generation size and number of swaps must be at least 1')
                return None
            if any(value > 1 for value in algo_values[:3])  :
                sg.popup('rates must be between 0 and 1')
                return None
            
            return algo_values
        
        except:
            sg.popup('Error parsing input')
            return None
                

        


    def start(self):
        """
        starts the main window in which the parameters are set,
        allows multiple simulations to run and to change the parameters
        for each one without having to close the program
        """
        self.window = sg.Window('Genetic Algorithms',
                                    self.main_layout,
                                    finalize=True,
                                    resizable=True,
                                    element_justification="left")
        # MAIN LOOP
        while True:
            event, values = self.window.read(timeout=200)
           
            if event == sg.WIN_CLOSED or event == 'Exit':
                break

            if event == 'Genetic Algorithm':
                # set algorithm type to cooldown, disable its button
                self.algorithm = 'Genetic Algorithm'
                self.window['Genetic Algorithm'].update(disabled=True)
                self.window['Darwin Variation'].update(disabled=False)
                self.window['Lamarck Variation'].update(disabled=False)

            if event == 'Darwin Variation':
                # set algorithm type to heard rumour, disable its button
                self.algorithm = 'Darwin Variation'
                self.window['Darwin Variation'].update(disabled=True)
                self.window['Genetic Algorithm'].update(disabled=False)
                self.window['Lamarck Variation'].update(disabled=False)

            if event == 'Lamarck Variation':
                # set algorithm type to got_rumour, disable its button
                self.algorithm = 'Lamarck Variation'
                self.window['Lamarck Variation'].update(disabled=True)
                self.window['Darwin Variation'].update(disabled=False)
                self.window['Genetic Algorithm'].update(disabled=False)

            if event == 'Show Graph':
                self.show_graph_flag = True
                self.window['Show Graph'].update(disabled=True)
                self.window['Dont Show Graph'].update(disabled=False)
            
            if event == 'Dont Show Graph':
                self.show_graph_flag = False
                self.window['Dont Show Graph'].update(disabled=True)
                self.window['Show Graph'].update(disabled=False)

            if event == 'Information':
                # start a popup window with information about the program
                sg.popup(self.infotext, font=self.AppFont)

            if event == 'Start Decoding':
                # process user entered parameters
                algo_values = self.process_values(values)

                # invalid values entered, error is printed in the function
                if algo_values == None:
                    continue
                
                self.run_algorithm(algo_values)
            
            if event == 'Generate Stats':
                # process user entered parameters
                algo_values = self.process_values(values)

                # invalid values entered, error is printed in the function
                if algo_values == None:
                    continue
                
                self.generate_stats(algo_values)
        
        try:
            os.remove("graph.png")
        except:
            pass
                
        self.window.close()