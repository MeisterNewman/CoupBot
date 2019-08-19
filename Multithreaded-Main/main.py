import os
import numpy as np

from communication import CommunicationChannel
import models

from time import sleep
import train



# We need 7 communications channels: one for evaluation data to the supervisor, one for evaluation results from the supervisor, and five for data to the individual trainers
##### We need 11 pipes per thread in total: one pipe to indicate that the supervisor needs to read data from the evaluation pipe for the nth evaluator, one to indicate that the supervisor has responded
##### the evaluation pipe to the supervisor, the evaluation pipe from the supervisor, the five training data pipes to the supervisor, one pipe to indicate which of the training data pipes are holding training data, and one pipe to count games played per thread.


import tensorflow as tf
import keras
from keras import backend as K



num_models = 5
import threading


class CentralTrainer:
    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        #K.manual_variable_initialization(True)
        self.networks = [models.get_action_evaluator(),  # These wrappers make predict functions as well, so no issue there
                    models.get_block_evaluator(steal=False),
                    models.get_block_evaluator(steal=False),
                    models.get_block_evaluator(steal=True),
                    models.get_challenge_evaluator()]

        self.train_queues = []
        self.train_queues_sample_counts = []
        self.lock_queues = []
        self.MIN_BATCH_SIZE = 2048
        self.games_played = 0
        self.games_played_lock = threading.Lock()

        for i in self.networks:
            self.train_queues += [[]]
            self.train_queues_sample_counts += [0]
            self.lock_queues += [threading.Lock()]

    def add_training_data(self, model_index, data):
        with self.lock_queues[model_index]:
            self.train_queues[model_index]+=[data]
            self.train_queues_sample_counts+=[data[1].shape[0]]

    def try_to_train(self, model_index):
        if self.train_queues_sample_counts[model_index] < self.MIN_BATCH_SIZE:
            sleep(0)
            return
        with self.lock_queues[model_index]:
            if self.train_queues_sample_counts[model_index] < self.MIN_BATCH_SIZE:
                sleep(0)
                return
            else:
                train_data = self.train_queues[model_index]
                self.train_queues[model_index] = []
                self.train_queues_sample_counts[model_index] = 0
        input = []
        for i in range (len(train_data[0][0])): # However many different network inputs there are
            input += np.concatenate([datum[0][i] for datum in train_data], axis=0)
        output = np.concatenate([datum[1] for datum in train_data], axis=0)

        self.networks[model_index].fit(input, output, batch_size=4096, epochs=1)

    def predict(self, model_index, data):
        return self.networks[model_index].predict(data)

    def increment_games_played(self):
        with self.games_played_lock:
            self.games_played += 1



class NetworkMask:
    def __init__(self, center, index):
        self.index = index
        self.center = center
    def fit(self, x, y, **kwargs):
        self.center.add_training_data(self.index, (x,y))
    def predict(self, x):
        return self.center.predict(self.index, x)

class GameThread(threading.Thread):
    stop_signal = False

    def __init__(self, center, num_players, epsilon):
        threading.Thread.__init__(self)

        self.center = center
        self.num_players = num_players
        self.epsilon = epsilon

        self.trainer_args = [self.num_players]
        for i in range(len(self.center.networks)):
            self.trainer_args+=[NetworkMask(self.center, i)]
        self.trainer_args += [self.epsilon, 0]  # 0 for verbose
    def run(self):
        while not self.stop_signal:
            trainer = train.GameTrainingWrapper(*self.trainer_args)
            game_continuing = True
            while game_continuing and not self.stop_signal:
                game_continuing = trainer.take_turn()
                #print("Turn taken")
            if not self.stop_signal:
                trainer.train_all_evaluators(verbose=0)  # Indent this for more frequent training
            self.center.increment_games_played()

    def stop(self):
        self.stop_signal = True

class ModelTrainer(threading.Thread):
    stop_signal = False

    def __init__(self, center, index):
        threading.Thread.__init__(self)
        self.index=index
        self.center = center

    def run(self):
        while not self.stop_signal:
            self.center.try_to_train(self.index)

    def stop(self):
        self.stop_signal = True


central_control = CentralTrainer()
NUM_THREADS = 64
EPSILON = .4

game_threads = [GameThread(central_control, 5, EPSILON) for i in range(4)]
optimizers = [ModelTrainer(central_control, index) for index in range(len(central_control.networks))]

for o in optimizers:
    o.start()

for g in game_threads:
    g.start()

for t in range(200):
    print ("Time: " + str(t) + " Games played: " + str(central_control.games_played))
    if len(game_threads)< NUM_THREADS:
        game_threads += [GameThread(central_control, 5, EPSILON)]
        game_threads[-1].start()
    sleep(1)

for g in game_threads:
    g.stop()
for g in game_threads:
    g.join()


trainer_args = [5]
for i in range(len(central_control.networks)):
    trainer_args+=[NetworkMask(central_control, i)]
trainer_args += [0, 1]  # 0 for verbose
trainer = train.GameTrainingWrapper(*trainer_args)
game_continuing=True
while game_continuing:
    game_continuing = trainer.take_turn()



for o in optimizers:
    o.stop()
for o in optimizers:
    o.join()












