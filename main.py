import os
import numpy as np

from communication import CommunicationChannel




# We need 7 communications channels: one for evaluation data to the supervisor, one for evaluation results from the supervisor, and five for data to the individual trainers
##### We need 11 pipes per thread in total: one pipe to indicate that the supervisor needs to read data from the evaluation pipe for the nth evaluator, one to indicate that the supervisor has responded
##### the evaluation pipe to the supervisor, the evaluation pipe from the supervisor, the five training data pipes to the supervisor, one pipe to indicate which of the training data pipes are holding training data, and one pipe to count games played per thread.

def concatenate_lists_of_arrays(l1, l2):
    # print([i.shape for i in l1], "\n", [i.shape for i in l2])
    return [np.concatenate([l1[i], l2[i]], axis=0) for i in range (len(l1))]





games_per_thread = 25
num_threads = 128
NUM_EVALUATORS = 5


#Each thread has one pair of channels for each model: one to send data to it, one to receive data from it.
thread_pipes = []
for i in range (num_threads+1):
    thread_pipes += [[]]
    for i in range (NUM_EVALUATORS):
        thread_pipes[-1] += [[CommunicationChannel(), CommunicationChannel()]]


game_thread = False
trainer_thread = False
model_pids = []

for i in range (NUM_EVALUATORS):
    model_pids+=[os.fork()]
    if model_pids[-1]==0:
        import models
        model_index = len(model_pids)-1
        trainer_thread = True
        thread_pipes = [i[model_index] for i in thread_pipes]
        model_data_pipes = thread_pipes

        if model_index == 0:
            model = models.get_action_evaluator()
        elif model_index == 1:
            model = models.get_block_evaluator(steal=False)
        elif model_index == 2:
            model = models.get_block_evaluator(steal=False)
        elif model_index == 3:
            model = models.get_block_evaluator(steal=True)
        elif model_index == 4:
            model = models.get_challenge_evaluator()
        else:
            raise RuntimeError("Too many models!")

        break






if trainer_thread:

    from time import perf_counter

    eval_stack = None
    eval_owner_stack = []
    training_data_x_stack = None
    training_data_y_stack = None

    min_length_table = {
        0: 36,
        1: 4,
        2: 4,
        3: 4,
        4: 2,
    }
    last_eval_time=perf_counter()
    greedy_eval=False
    while 1:
        for i in range (len(model_data_pipes)):
            if model_data_pipes[i][0].has_data():
                if model_data_pipes[i][0].read()=="t": # If the data is training data, add it to the stacks
                    data = model_data_pipes[i][0].read()
                    if training_data_x_stack is None:
                        training_data_x_stack = data[0]
                        training_data_y_stack = data[1]
                    else:
                        training_data_x_stack = models.concatenate_lists_of_arrays(training_data_x_stack, data[0])
                        training_data_y_stack = np.concatenate([training_data_y_stack, data[1]], axis=0)
                else:
                    data = model_data_pipes[i][0].read()  # It's evaluation data
                    if eval_stack is None:  # If the stack is empty, set it to this data
                        eval_stack = data
                    else:  # Otherwise, concatenate this data on. We have a list of input arrays, each of which must concatenate individually
                        eval_stack = models.concatenate_lists_of_arrays(eval_stack, data)
                    eval_owner_stack += [[i, data[0].shape[0]]]

            if (len(eval_owner_stack)>=min_length_table[model_index]) or (greedy_eval and len(eval_owner_stack)>0):
                #print("Started eval")
                eval_result = model.predict(eval_stack, verbose=0, batch_size=4096)
                #print("Finished eval")
                stack_index = 0
                for i in range (len(eval_owner_stack)):
                    model_data_pipes[eval_owner_stack[i][0]][1].write(eval_result[stack_index:stack_index+eval_owner_stack[i][1]])
                    stack_index += eval_owner_stack[i][1]
                assert stack_index == eval_result.shape[0]
                eval_stack = None
                eval_owner_stack = []
                last_eval_time=perf_counter()

        if not (training_data_y_stack is None) and (training_data_y_stack.shape[0]>2048):
            model.fit(training_data_x_stack, training_data_y_stack, batch_size=4096, epochs=1, verbose=0, shuffle=False, )
            training_data_x_stack = None
            training_data_y_stack = None
        if perf_counter() - last_eval_time > 10:
            greedy_eval = True



else: #If we are not a trainer thread, we are still the top thread: set up game threads now.
    class ModelRequestWrapper:
        def __init__(self, eval_in_channel, eval_out_channel):
            self.eval_in_channel = eval_in_channel
            self.eval_out_channel = eval_out_channel

        def fit(self, x, y, **kwargs):
            self.eval_in_channel.write("t")
            self.eval_in_channel.write((x, y))

        def predict(self, x, **kwargs):
            #print("Started request")
            self.eval_in_channel.write("p")
            self.eval_in_channel.write(x)
            self.eval_out_channel.idle_until_data()
            data = self.eval_out_channel.read()
            #print("Finished request")
            return data


    my_index = -1
    game_pids = []
    for i in range(num_threads):
        game_pids += [os.fork()]
        if game_pids[-1] == 0:  # If we are the child thread:
            my_index = len(game_pids) - 1
            game_thread = True
            thread_pipes = thread_pipes[my_index]
            my_pipes = thread_pipes
            break

    if game_thread:
        import train
        evaluators = []
        for i in range (NUM_EVALUATORS): #Generate the five evaluators. They will communicate with the parent process for direction
            evaluators += [ModelRequestWrapper(my_pipes[i][0], my_pipes[i][1])]
        action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator = evaluators #Map them in

        for i in range (games_per_thread):
            if my_index==0:
                print("Playing game", i)
            trainer = train.GameTrainingWrapper(5, action_evaluator, assassin_block_evaluator, aid_block_evaluator,
                                                captain_block_evaluator, challenge_evaluator, q_epsilon=.4, verbose=False)
            game_continuing = True
            while game_continuing:
                game_continuing = trainer.take_turn()
            trainer.train_all_evaluators(verbose=0)  # Indent this for more frequent training
        os._exit(0)


    else:  # If we are the central supervisory process, wait for all game-playing threads to terminate.
        print ("All children successfully started.")
        num_threads_running = num_threads
        my_pipes = thread_pipes[-1]
        del thread_pipes
        for i in range (num_threads):
            if os.waitpid(game_pids[i], 0) != (0,0): # could do os.WNOHANG
                num_threads_running-=1

        evaluators = []
        for i in range(NUM_EVALUATORS):  # Generate the evaluators. They will communicate with the parent process for direction
            evaluators += [ModelRequestWrapper(my_pipes[i][0], my_pipes[i][1])]
        action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator = evaluators  # Map them in

        import train
        trainer = train.GameTrainingWrapper(5, action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator, q_epsilon=0, verbose=True)
        game_continuing=True
        while game_continuing:
            game_continuing = trainer.take_turn()

        for p in model_pids:
            os.kill(p, 15) #SIGTERM all model processes