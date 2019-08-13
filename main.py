import os
import numpy as np

from communication import CommunicationChannel




# We need 7 communications channels: one for evaluation data to the supervisor, one for evaluation results from the supervisor, and five for data to the individual trainers
##### We need 11 pipes per thread in total: one pipe to indicate that the supervisor needs to read data from the evaluation pipe for the nth evaluator, one to indicate that the supervisor has responded
##### the evaluation pipe to the supervisor, the evaluation pipe from the supervisor, the five training data pipes to the supervisor, one pipe to indicate which of the training data pipes are holding training data, and one pipe to count games played per thread.

def concatenate_lists_of_arrays(l1, l2):
    # print([i.shape for i in l1], "\n", [i.shape for i in l2])
    return [np.concatenate([l1[i], l2[i]], axis=0) for i in range (len(l1))]





games_per_thread = 7000
num_threads = 15

pids = []
thread_pipes = []

supervisor_process = True

my_index = -1
for i in range (num_threads):
    thread_pipes+=[[]]
    for j in range (7):
        thread_pipes[-1]+=[CommunicationChannel()]
    pids+=[os.fork()]
    if pids[-1]==0: #If we are the child thread:
        my_index=len(thread_pipes)-1
        supervisor_process = False
        my_pipes=thread_pipes.pop(-1)

        break




if supervisor_process: # If we are not the supervisor process, begin running games.
    import models

    action_evaluator = models.get_action_evaluator()
    assassin_block_evaluator = models.get_block_evaluator(steal=False)
    aid_block_evaluator = models.get_block_evaluator(steal=False)
    captain_block_evaluator = models.get_block_evaluator(steal=True)
    challenge_evaluator = models.get_challenge_evaluator()

    evaluators = [
        action_evaluator,
        assassin_block_evaluator,
        aid_block_evaluator,
        captain_block_evaluator,
        challenge_evaluator,
    ]

    num_threads_running = num_threads

    model_eval_stacks = [None, None, None, None, None]
    model_eval_owner_stacks = [[], [], [], [], []]
    threads_waiting_for_eval = []
    thread_running = []
    for i in range (num_threads):
        threads_waiting_for_eval += [False]
        thread_running += [True]

    training_data_stacks = [[None, None], [None, None], [None, None], [None, None], [None, None]]

    cycle = 0
    while num_threads_running>0:
        for thread in range (num_threads):  #Go through all threads, checking to see if they have data that needs to be evaluated. If so, add it to the evaluation stacks
            if thread_running[thread] and not threads_waiting_for_eval[thread]:
                if thread_pipes[thread][0].has_data():
                    index=thread_pipes[thread][0].read()
                    data=thread_pipes[thread][0].read()
                    if type(model_eval_stacks[index])==type(None):  #If the stack is empty, set it to this data
                        model_eval_stacks[index] = data
                    else: #Otherwise, concatenate this data on. We have a list of input arrays, each of which must concatenate individually
                        model_eval_stacks[index] = concatenate_lists_of_arrays(model_eval_stacks[index], data)
                    model_eval_owner_stacks[index] += [[thread, data[0].shape[0]]]

                    threads_waiting_for_eval[thread] = True
        if cycle % 1 == 0:
            for q in range (1):
                queue_depths = [len(i) for i in model_eval_owner_stacks]
                deepest_evaluation_queue = queue_depths.index(max(queue_depths))
                if queue_depths[deepest_evaluation_queue]>0:  # If there is something in the maximal evaluation queue
                    evaluation_result = evaluators[deepest_evaluation_queue].predict(model_eval_stacks[deepest_evaluation_queue], verbose=0)
                    stack_index=0
                    for i in range (len(model_eval_owner_stacks[deepest_evaluation_queue])):  # Write back the results to each thread
                        thread=model_eval_owner_stacks[deepest_evaluation_queue][i][0]
                        thread_pipes[thread][1].write(evaluation_result[stack_index:stack_index+model_eval_owner_stacks[deepest_evaluation_queue][i][1]])
                        stack_index+=model_eval_owner_stacks[deepest_evaluation_queue][i][1]
                        threads_waiting_for_eval[thread] = False
                    assert stack_index == evaluation_result.shape[0]
                    model_eval_stacks[deepest_evaluation_queue]=None
                    model_eval_owner_stacks[deepest_evaluation_queue]=[]

            for thread in range (num_threads):
                if thread_running[thread] and not threads_waiting_for_eval[thread]:
                    for index in range (0,5):
                        if thread_pipes[thread][2+index].has_data():
                            data = thread_pipes[thread][2+index].read()
                            # print(data)
                            if training_data_stacks[index][0] is None:
                                training_data_stacks[index][0] = data[0]
                                training_data_stacks[index][1] = data[1]
                            else:
                                training_data_stacks[index][0] = concatenate_lists_of_arrays(training_data_stacks[index][0], data[0])
                                training_data_stacks[index][1] = np.concatenate([training_data_stacks[index][1], data[1]], axis=0)

            for i in range (len(training_data_stacks)):  # For each stack, check if it has enough data to train. If so, train it
                if type(training_data_stacks[i][1])!=type(None) and training_data_stacks[i][1].shape[0] > 256:
                    evaluators[i].fit(training_data_stacks[i][0], training_data_stacks[i][1], batch_size=32, epochs=1, verbose=0)
                    training_data_stacks[i]=[None, None]
        if cycle%100 == 0:
            for i in range (num_threads):
                if thread_running[i]:
                    if os.waitpid(pids[i], os.WNOHANG) != (0,0):
                        thread_running[i]=False
                        num_threads_running-=1

        cycle+=1

else:
    import train
    class ModelRequestWrapper:
        def __init__(self, eval_in_channel, eval_out_channel, training_data_channel, flag):
            self.eval_in_channel = eval_in_channel
            self.eval_out_channel = eval_out_channel
            self.training_data_channel = training_data_channel
            self.flag=flag

        def fit(self, x, y, **kwargs):
            self.training_data_channel.write((x,y))

        def predict(self, x, **kwargs):
            self.eval_in_channel.write(self.flag)
            self.eval_in_channel.write(x)
            self.eval_out_channel.idle_until_data()
            data=self.eval_out_channel.read()
            return data

    evaluators=[]

    for i in range (5): #Generate the five evaluators. They will communicate with the parent process for direction
        evaluators += [ModelRequestWrapper(my_pipes[0], my_pipes[1], my_pipes[2+i], i)]
    action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator = evaluators #Map them in
    #print(action_evaluator)
    #print(assassin_block_evaluator)
    #print(aid_block_evaluator)
    #print(captain_block_evaluator)
    #print(challenge_evaluator)

    for i in range (games_per_thread):
        if my_index==0:
            print("Playing game", i)
        trainer = train.GameTrainingWrapper(5, action_evaluator, assassin_block_evaluator, aid_block_evaluator,
                                            captain_block_evaluator, challenge_evaluator)
        game_continuing = True
        while game_continuing:
            game_continuing = trainer.take_turn()
        trainer.train_all_evaluators(verbose=0)  # Indent this for more frequent training

    os._exit(0)



import train
trainer = train.GameTrainingWrapper(5, action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator)
game_continuing=True
while game_continuing:
    game_continuing = trainer.take_turn(verbose=True)


# import cProfile
#
# def f():
#     for i in range(1000):
#         if i % 100 == 0:
#             print(i)
#         trainer = train.GameTrainingWrapper(5, action_evaluator, assassin_block_evaluator, aid_block_evaluator,
#                                             captain_block_evaluator, challenge_evaluator)
#         game_continuing = True
#         while game_continuing:
#             game_continuing = trainer.take_turn()
#         trainer.train_all_evaluators(verbose=0)
#
# cProfile.run("f()")
