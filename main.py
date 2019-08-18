import os
import numpy as np

from communication import CommunicationChannel

from copy import deepcopy


# We need 7 communications channels: one for evaluation data to the supervisor, one for evaluation results from the supervisor, and five for data to the individual trainers
##### We need 11 pipes per thread in total: one pipe to indicate that the supervisor needs to read data from the evaluation pipe for the nth evaluator, one to indicate that the supervisor has responded
##### the evaluation pipe to the supervisor, the evaluation pipe from the supervisor, the five training data pipes to the supervisor, one pipe to indicate which of the training data pipes are holding training data, and one pipe to count games played per thread.

def concatenate_lists_of_arrays(l1, l2):
    # print([i.shape for i in l1], "\n", [i.shape for i in l2])
    return [np.concatenate([l1[i], l2[i]], axis=0) for i in range (len(l1))]





games_per_thread = 1000000000  # Outdated
runtime = 60  # In seconds
num_threads = 128
NUM_EVALUATORS = 6


#Each thread has one pair of channels for each model: one to send data to it, one to receive data from it.
thread_pipes = []
for i in range (num_threads+1):
    thread_pipes += [[]]
    for i in range (NUM_EVALUATORS):
        thread_pipes[-1] += [[CommunicationChannel(), CommunicationChannel()]]


game_thread = False
trainer_thread = False
model_pids = []

manager_pid = os.getpid()


for i in range (NUM_EVALUATORS):
    model_pids+=[os.fork()]
    if model_pids[-1]==0:
        trainer_thread = True
        model_index = len(model_pids) - 1
        break






if trainer_thread:  # We fork every training thread into two components: scanner and evaluator

    trainer_internal_pipes = [CommunicationChannel(), CommunicationChannel()]  # First is from the gatherer to the trainer, second is vice versa
    thread_pipes = [i[model_index] for i in thread_pipes]
    model_data_pipes = thread_pipes
    runner_pid = os.fork()
    if runner_pid == 0:  #If we are the actual trainer/evaluator:
        import models
        model_index = len(model_pids) - 1
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
        elif model_index == 5:
            model = models.get_game_state_predictor()
        else:
            raise RuntimeError("Too many models!")

        while 1:
            trainer_internal_pipes[0].idle_until_data()
            ins = trainer_internal_pipes[0].read()
            if ins == "t":
                data = trainer_internal_pipes[0].read()
                model.fit(data[0], data[1], batch_size=4096, epochs=1, verbose=0, shuffle=False, )
            else:
                data, eval_owner_stack = trainer_internal_pipes[0].read()
                eval_result = model.predict(data, verbose=0, batch_size=data[0].shape[0])
                stack_index = 0
                for i in range(len(eval_owner_stack)):
                    model_data_pipes[eval_owner_stack[i][0]][1].write(
                        eval_result[stack_index:stack_index + eval_owner_stack[i][1]] )
                    stack_index += eval_owner_stack[i][1]
                assert stack_index == eval_result.shape[0]
            trainer_internal_pipes[1].write("d")

    else:
        from time import perf_counter
        last_eval_time = perf_counter()


        num_pipes = len(model_data_pipes)

        min_length_table = {  # The sum of these must be less than the total number of threads
            0: 25,  # This and 5 may want to be changed
            1: 4,  # 4
            2: 4,  # 4
            3: 4,  # 4
            4: 5,  # 2
            5: 25,
        }

        min_length_table_2 = {  # The sum of these must be less than the total number of threads
            0: 25,  # This and 5 may want to be changed
            1: 4,  # 4
            2: 4,  # 4
            3: 4,  # 4
            4: 5,  # 2
            5: 25,
        }

        min_train_table = {
            0: 2048,
            1: 512,
            2: 512,
            3: 512,
            4: 2048,
            5: 2048,
        }

        eval_stack = []
        eval_owner_stack = []
        training_data_x = []
        training_data_y = []

        training_samples = 0

        can_send = True
        while 1:
            for i in range(num_pipes):
                if model_data_pipes[i][0].has_data():
                    ins = model_data_pipes[i][0].read()
                    if ins == "t":  # If the data is training data, add it to the stacks
                        data = model_data_pipes[i][0].read()
                        training_data_x += [data[0]]
                        training_data_y += [data[1]]
                        training_samples += data[1].shape[0]
                    elif ins == "p":
                        data = model_data_pipes[i][0].read()  # It's evaluation data
                        eval_stack += [data]
                        eval_owner_stack += [[i, data[0].shape[0]]]
                    else:  # We must both fit to and predict on this data
                        data = model_data_pipes[i][0].read()
                        eval_stack += [data[0]]
                        eval_owner_stack += [[i, data[0][0].shape[0]]]
                        training_data_x += [data[0]]
                        training_data_y += [data[1]]
                        training_samples += data[1].shape[0]

                if can_send:
                    if len(eval_owner_stack) > 0 and perf_counter()-last_eval_time>.002:  # and (model_index != 5 or len(eval_owner_stack)>40):
                        last_eval_time=perf_counter()
                        model_in = []
                        for d in range(len(eval_stack[0])):
                            model_in += [np.concatenate([e[d] for e in eval_stack], axis=0)]
                        trainer_internal_pipes[0].write("e")
                        trainer_internal_pipes[0].write((model_in, eval_owner_stack))

                        eval_stack = []
                        del model_in
                        eval_owner_stack = []

                        can_send = False
                    elif (training_samples > min_train_table[model_index]):
                        train_in = []
                        for d in range(len(training_data_x[0])):
                            train_in += [np.concatenate([e[d] for e in training_data_x], axis=0)]
                        train_out = np.concatenate(training_data_y, axis=0)

                        trainer_internal_pipes[0].write("t")
                        trainer_internal_pipes[0].write((train_in, train_out))
                        training_data_x = []
                        training_data_y = []
                        del train_in
                        del train_out
                        training_samples = 0
                        can_send = False

                else:
                    if trainer_internal_pipes[1].has_data():
                        trainer_internal_pipes[1].read()
                        can_send = True



            try:  # If the parent isn't running, die.
                os.kill(manager_pid, 0)
            except:
                os.kill(runner_pid, 15)
                exit(0)

    # from time import perf_counter
    #
    # last_eval_time = perf_counter()
    # num_pipes = len(model_data_pipes)
    #
    #
    # while 1:
    #     for i in range (num_pipes):
    #         if model_data_pipes[i][0].has_data():
    #             ins = model_data_pipes[i][0].read()
    #             if ins=="t": # If the data is training data, add it to the stacks
    #                 data = model_data_pipes[i][0].read()
    #                 training_data_x += [data[0]]
    #                 training_data_y += [data[1]]
    #                 training_samples += data[1].shape[0]
    #             elif ins=="p":
    #                 data = model_data_pipes[i][0].read()  # It's evaluation data
    #                 eval_stack += [data]
    #                 eval_owner_stack += [[i, data[0].shape[0]]]
    #             else: #We must both fit to and predict on this data
    #                 data = model_data_pipes[i][0].read()
    #                 eval_stack += [data[0]]
    #                 eval_owner_stack += [[i, data[0][0].shape[0]]]
    #                 training_data_x += [data[0]]
    #                 training_data_y += [data[1]]
    #                 training_samples += data[1].shape[0]
    #
    #             if ((len(eval_owner_stack) >= min_length_table[model_index]) or len(eval_owner_stack) > 0 and perf_counter() - last_eval_time > .1):
    #                 last_eval_time = perf_counter()
    #
    #                 model_in = []
    #                 for d in range (len(eval_stack[0])):
    #                     model_in += [np.concatenate([e[d] for e in eval_stack], axis=0)]
    #                 eval_result = model.predict(model_in, verbose=0, batch_size=eval_stack[0][0].shape[0])
    #                 stack_index = 0
    #                 for i in range(len(eval_owner_stack)):
    #                     model_data_pipes[eval_owner_stack[i][0]][1].write(
    #                         eval_result[stack_index:stack_index + eval_owner_stack[i][1]])
    #                     stack_index += eval_owner_stack[i][1]
    #                 assert stack_index == eval_result.shape[0]
    #                 eval_stack = []
    #                 del model_in
    #                 eval_owner_stack = []
    #
    #
    #                 if (training_samples > min_train_table[model_index]):
    #                     train_in = []
    #                     for d in range(len(training_data_x[0])):
    #                         train_in += [np.concatenate([e[d] for e in training_data_x], axis=0)]
    #                     train_out = np.concatenate(training_data_y, axis=0)
    #
    #                     model.fit(train_in, train_out, batch_size=4096, epochs=1, verbose=0,
    #                               shuffle=False, )
    #                     training_data_x = []
    #                     training_data_y = []
    #                     del train_in
    #                     del train_out
    #                     training_samples = 0
    #
    #
    #     try:  # If the parent isn't running, die.
    #         os.kill(manager_pid, 0)
    #     except:
    #         exit(0)



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
            return self.eval_out_channel.read()

        def fit_predict(self, x, y, **kwargs):
            self.eval_in_channel.write("b")
            self.eval_in_channel.write((x,y))
            self.eval_out_channel.idle_until_data()
            return self.eval_out_channel.read()


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
        from time import perf_counter
        last_eval_time = perf_counter()
        import train
        evaluators = []
        for i in range (NUM_EVALUATORS): #Generate the five evaluators. They will communicate with the parent process for direction
            evaluators += [ModelRequestWrapper(my_pipes[i][0], my_pipes[i][1])]
        action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator, game_state_evaluator = evaluators #Map them in

        for i in range (games_per_thread):
            eps = .4*(.999**i)
            if my_index==16:
                print("Playing game", i, "with epsilon", eps, " . Time for previous hand:", perf_counter()-last_eval_time)
                last_eval_time=perf_counter()
            trainer = train.GameTrainingWrapper(5, action_evaluator, assassin_block_evaluator, aid_block_evaluator,
                                                captain_block_evaluator, challenge_evaluator, game_state_evaluator,
                                                q_epsilon=eps, verbose=False)
            game_continuing = True
            while game_continuing:
                game_continuing = trainer.take_turn()
            trainer.train_all_evaluators(verbose=0)  # Indent this for more frequent training
        os._exit(0)


    else:  # If we are the central supervisory process, wait for all game-playing threads to terminate.
        from time import sleep
        print ("All children successfully started.")
        num_threads_running = num_threads
        my_pipes = thread_pipes[-1]
        del thread_pipes

        sleep(runtime)

        for i in game_pids:
            os.kill(i, 15)
            os.wait()

        evaluators = []
        for i in range(NUM_EVALUATORS):  # Generate the evaluators. They will communicate with the parent process for direction
            evaluators += [ModelRequestWrapper(my_pipes[i][0], my_pipes[i][1])]
        action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator, game_state_evaluator = evaluators  # Map them in

        import train
        trainer = train.GameTrainingWrapper(5, action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator, game_state_evaluator, q_epsilon=0, verbose=True)
        game_continuing=True
        while game_continuing:
            game_continuing = trainer.take_turn()

        for p in model_pids:
            os.kill(p, 15) #SIGTERM all model processes
            os.wait()