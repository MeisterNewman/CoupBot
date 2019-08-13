import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import keras
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Lambda
from keras import backend as K
K.set_floatx('float32')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))




import game

import numpy as np
def concatenate_lists_of_arrays(l1, l2):
    # print([i.shape for i in l1], "\n", [i.shape for i in l2])
    return [np.concatenate([l1[i], l2[i]], axis=0) for i in range (len(l1))]






#TODO: Make ambassador choice-making mechanism

def stackLayers(layerSet):
    stack = layerSet[0]
    for i in range(1, len(layerSet)):
        stack = layerSet[i](stack)
    return stack



#First player in order will be acting player, last in order will be us. Any players not in game have input as pure zeros


def get_game_state_predictor(): #Generate a network to predict state of cards in action-taking player's hand, given previous state of game. This intentionally does not use our knowledge of our own cards -- this makes training far faster and is useful when we decide whether to block
    undiscarded_cards = Input(shape=(5,)) #How many of each card remains in play


    prior_probability_input = Input(shape=(6,5)) #Up to 6 players, and the predicted probabilities of their five cards. Acting player's first

    num_cards = Input(shape=(6,))  # How many cards each player has (acting player's first)
    num_coins = Input(shape=(6,))  # How many coins each player has (acting player's first)

    action_input = Input(shape=(game.NUM_ACTIONS,)) #One-hot, with the exception of a FAILURE to block: in this case, the relevant card category should be made -1
    layers=[
        keras.layers.concatenate([undiscarded_cards,
                                  Flatten()(prior_probability_input),
                                  Lambda(lambda x: x/2)(num_cards),
                                  Lambda(lambda x: x/10)(num_coins),
                                  action_input]),

        Dense(128, activation='relu'),
        Dropout(.3),
        Dense(128, activation='relu'),
        Dropout(.3),
        Dense(64, activation='relu'),
        Dropout(.3),
        Dense(32, activation='relu'),
        Dropout(.3),
        Dense(16, activation='relu'),
        Dropout(.3),
        Dense(5, activation='sigmoid'),
        Lambda(lambda x: x * 2.0),
    ]
    net = keras.models.Model(inputs=(undiscarded_cards,
                                     prior_probability_input,
                                     num_cards,
                                     num_coins,
                                     action_input),
                             outputs=stackLayers(layers))
    net.compile(optimizer=keras.optimizers.Adam(.003), loss='mse', metrics=['accuracy'])
    return net

def get_action_evaluator():#Generates a network to decide the value of an action, given game state.    May want to make it a conv 1d net for the prior probability input
    undiscarded_cards = Input(shape=(5,))  # How many of each card remains in play


    prior_probability_input = Input(shape=(6,5))  # Up to 6 players, and the predicted probabilities of their five cards. We are first.

    our_actual_cards = Input(shape=(5,)) #One-hot, with a 2 if we have the same
    num_cards = Input(shape=(6,))  # How many cards each player has (ours first)
    num_coins = Input(shape=(6,))  # How many coins each player has (ours first)

    random_noise = Input(shape=(5,)) #Random noise for decision-making

    action = Input(shape=(game.NUM_ACTIVE_ACTIONS,))
    target = Input(shape=(game.MAX_PLAYERS-1,)) #Our target among the other 5 players. Other players should come in same order as above.


    layers = [
        keras.layers.concatenate([undiscarded_cards,
                                  our_actual_cards,
                                  Flatten()(prior_probability_input),
                                  Lambda(lambda x: x / 2)(num_cards),
                                  Lambda(lambda x: x / 10)(num_coins),
                                  random_noise,
                                  action,
                                  target]),
        Dense(128, activation='relu'),
        Dropout(.3),
        Dense(128, activation='relu'),
        Dropout(.3),
        Dense(64, activation='relu'),
        Dropout(.3),
        Dense(32, activation='relu'),
        Dropout(.3),
        Dense(8, activation='relu'),
        Dropout(.3),
        Dense(1, activation='sigmoid'),

    ]

    net = keras.models.Model(inputs=(undiscarded_cards,
                                     our_actual_cards,
                                     prior_probability_input,
                                     num_cards,
                                     num_coins,
                                     random_noise,
                                     action,
                                     target),
                             outputs=stackLayers(layers))
    net.compile(optimizer=keras.optimizers.Adam(.003), loss='mse', metrics=['accuracy'])
    return net

def get_block_evaluator(steal): #Generates reward evaluator for a specific blocking action. Set steal=true iff that action is stealing
    undiscarded_cards = Input(shape=(5,))  # How many of each card remains in play

    prior_probability_input = Input(shape=(6, 5))  # The outside-predicted values of all cards (ours first, aggressor's second)
    our_actual_cards = Input(shape=(5,)) #One-hot, with a 2 if we have the same

    num_cards = Input(shape=(6,))  # How many cards each other player has
    num_coins = Input(shape=(6,))  # How many coins each other player has

    random_noise = Input(shape=(5,))  # Random noise for decision-making

    action=Input(shape=((2 if steal else 1),)) #Action to take. Note that if the action is blocking a steal, we have a one-hot encoding for which steal we do

    layers = [
        keras.layers.concatenate([undiscarded_cards,
                                  our_actual_cards,
                                  Flatten()(prior_probability_input),
                                  Lambda(lambda x: x / 2)(num_cards),
                                  Lambda(lambda x: x / 10)(num_coins),
                                  random_noise,
                                  action]),
        Dense(64, activation='relu'),
        Dropout(.3),
        Dense(32, activation='relu'),
        Dropout(.3),
        Dense(8, activation='relu'),
        Dropout(.3),
        Dense(1, activation='sigmoid'),
    ]
    net = keras.models.Model(inputs=(undiscarded_cards,
                                     our_actual_cards,
                                     prior_probability_input,
                                     num_cards,
                                     num_coins,
                                     random_noise,
                                     action),
                             outputs=stackLayers(layers))
    net.compile(optimizer=keras.optimizers.Adam(.003), loss='mse', metrics=['accuracy'])
    return net



def get_challenge_evaluator(): #Generates reward evaluator for challenges.
    undiscarded_cards = Input(shape=(5,))  # How many of each card remains in play

    prior_probability_input = Input(
        shape=(6, 5))  # The outside-predicted values of all cards (ours first, aggressor's second)
    our_actual_cards = Input(shape=(5,))  # One-hot, with a 2 if we have the same

    num_cards = Input(shape=(6,))  # How many cards each other player has
    num_coins = Input(shape=(6,))  # How many coins each other player has

    random_noise = Input(shape=(5,))  # Random noise for decision-making
    challengable_input = Input(shape=(game.NUM_CHALLENGABLE_ACTIONS,))  # One-hot -- what they did

    challenge = Input(shape=(1,)) #

    layers = [
        keras.layers.concatenate([undiscarded_cards,
                                  our_actual_cards,
                                  Flatten()(prior_probability_input),
                                  Lambda(lambda x: x / 2)(num_cards),
                                  Lambda(lambda x: x / 10)(num_coins),
                                  random_noise,
                                  challengable_input,
                                  challenge]),
        Dense(128, activation='relu'),
        Dropout(.3),
        Dense(128, activation='relu'),
        Dropout(.3),
        Dense(64, activation='relu'),
        Dropout(.3),
        Dense(8, activation='relu'),
        Dropout(.3),
        Dense(1, activation='sigmoid'),
    ]

    net = keras.models.Model(inputs=(undiscarded_cards,
                                     our_actual_cards,
                                     prior_probability_input,
                                     num_cards,
                                     num_coins,
                                     random_noise,
                                     challengable_input,
                                     challenge),
                             outputs=stackLayers(layers))
    net.compile(optimizer=keras.optimizers.Adam(.003), loss='mse', metrics=['accuracy'])
    return net


























