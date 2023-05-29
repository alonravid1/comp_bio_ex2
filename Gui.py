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

        # set main window's layout
        self.main_layout = [
            [sg.Button('Information', font=self.AppFont)],
            # set parameters
            [sg.Text('Enter parameters:', font='Any 18')],
            [sg.Text('Generation Size:',font=self.AppFont), sg.Input(key='gen_size', size=(15,1), font=self.AppFont, default_text='100')],
            [sg.Text('Replication Rate:',font=self.AppFont), sg.Input(key='replication_rate', size=(15,1), font=self.AppFont, default_text='0.3'),
             sg.Text('Mutation Rate:',font=self.AppFont), sg.Input(key='mutation_rate', size=(15,1), font=self.AppFont, default_text='0.3'),
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

            # run different kinds of simulation
            [sg.Button('Start Decoding', font=self.AppFont)],

            [sg.Button('Exit', font=self.AppFont)]
            ]
        
        self.infotext = "placeholder"

        """Welcome to the rumour spreading simulator.

        Please enter the following parameters as follows:
        * Generation Size
        * Replication Rate
        * Crossover Rate
        * Mutation Rate
        * Number of Mutations
        * Score Weights
        * Algorithm Type

        Visualisation types:
        * Genetic Algorithm - Color cells by how many iterations remain until it can spread the rumour again. A cell which spreads the rumour becomes bright and fades until it can spread it again.
        * Darwin Variation - Colors cells by whether or not they have heard the rumour.
        * Lamarck Variation - Colors cells by how many times they have heard the rumour throughout the simulation.
        * None - the simulation will run in the background and only present the last frame when it finishes.
        Generate Statistics:
        Set the number of repetitions, the simulation the runs for that number of times without visualisation, and then writes the average spread in a popup window.

        Strategic Simulation:
        Runs the simulation with a predefined algo.
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
                    [sg.Text(key="show_iter", font='any 18')],
                    [sg.Text(key="show_stats", font='any 18')],
                    [sg.Text(key="show_mess", font='any 18')]
                    ]
        
        return fresh_layout
        
    def draw_frame(self, window, stats):
        """
        function resizes the given frame to be bigger, saves it
        as an image and then shows it on the GUI for 0.05 of a second

        Args:
            window (sg.Window): a pysimplegui window
            garph (plt.figure): a graph

        """        
        avg_score_data = stats['avg']
        max_score_data = stats['max']
        iterations = np.arange(stats.size)
        plt.plot(iterations, avg_score_data, color='g', label="average score")
        plt.plot(iterations, max_score_data, color='r', label="max score")
        plt.legend()
        plt.title(f"{self.algorithm}")

        plt.savefig("frame.png")
        plt.clf()       

        try:
            window['frame'].update("frame.png")
            # set window to middle of screen
            screen_width, screen_height = window.get_screen_dimensions()
            win_width, win_height = window.size
            x, y = (screen_width - win_width)//2, (screen_height - win_height)//2
            window.move(x, y)
        except:
            # delete frame file
            try:
                os.remove("frame.png")
            except:
                pass
    
    def get_files(self):
        letter_freq = dict()
        pair_freq = dict()

        with open("enc.txt") as encrypted_file:
            enc_mess = encrypted_file.read()

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

        window = sg.Window('Evolution Presention',
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
            print_message = ""
            print_stats = ""
            algo.init_run(150)

            while not finished_flag:
                returned_stats = algo.iterate_step()
                solution, fitness_count, stats = returned_stats[:3]
                iteration, coverage, finished_flag = returned_stats[3:]

                plain_text = algo.decode_message(solution, enc_mess)
                print_decoded_mess = plain_text[0:98]

                if fitness_count == -1:
                    print_message = f"reset at iteration {iteration}"
                    window['show']
                elif not finished_flag:
                    print_stats=  f"generation: {iteration}, max score: {stats}, max coverage: {coverage}"
                    #print_decoded_mess

                event, values = window.read(timeout=2)
                if event == sg.WIN_CLOSED:
                    return
                
            print_message = f"best solution found at generation {iteration}, in {fitness_count} evaluations"
            print_stats = f"max score: {stats}, max coverage: {coverage}"
            # print_decoded_mess

            
            


            alphabet = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])

            with open("perm.txt", 'w+') as gen_perm:
                for i in range(len(solution)):
                    letter = alphabet[solution[i]]
                    gen_perm.write(f"{alphabet[i]} {letter}\n")

            with open("plain.txt", 'w+') as gen_sol:
                gen_sol.write(plain_text)

            sg.popup(f"Message decoded in {fitness_count} evaluations")


        
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
                
                
        self.window.close()