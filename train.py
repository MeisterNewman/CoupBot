import game
import numpy as np
from random import random, shuffle, randrange





def row_to_first(arr,row): #Given a given index in a numpy array, return a copy of the array with that index first (moving all between it and first in the process)
    return np.roll(arr,-row, axis=0)

def rows_to_first_second(arr, row1, row2):
    return row_to_first(row_to_first(arr, row2), row1 if row1>row2 else row1+1)


def zero_axis_tile(arr,num): #Gives an array of num elements, each of whose elements is a copy of the given array. Useful for expanding repeated training data
    return np.repeat(np.expand_dims(arr,axis=0), num, axis=0)

def append_slice(arr,slice):
    return np.append(arr, np.expand_dims(slice,axis=0), axis=0)

def one_hot(arr,num_cats):
    a=np.zeros((arr.shape[0],num_cats), dtype=np.float32)
    a[np.arange(arr.shape[0]), arr.astype(np.int)]=1
    return a



class ActionEvaluatorQueue: #This allows us to give action evaluation inputs and desired outputs at separate times
    def __init__(self):
        self.inputs = None
        self.output = None
    def append_inputs(self, inputs):
        if type(self.inputs)==type(None):
            self.inputs = [np.expand_dims(x,axis=0) for x in list(inputs)]
        else:
            self.inputs = [append_slice(self.inputs[i],inputs[i]) for i in range (0,len(inputs))]
    def append_output(self, output):
        if type(self.output)==type(None):
            self.output = np.expand_dims(output, axis=0)
        else:
            self.output = append_slice(self.output,output)

    def num_inputs(self):
        if type(self.inputs)==type(None):
            return 0
        return self.inputs[0].shape[0]
    def num_outputs(self):
        if type(self.output)==type(None):
            return 0
        return self.output.shape[0]

    def read_complete_data(self):
        if type(self.output)==type(None):
            return -1
        num_valid = self.output.shape[0]
        assert self.inputs[0].shape[0] >= num_valid
        rets = ([i[0:num_valid] for i in self.inputs], self.output[0:num_valid])
        self.inputs=self.inputs[num_valid:]
        self.output=None
        return rets

def combine_ready_from_list(queue_list):
    read=[x.read_complete_data() for x in queue_list]
    while -1 in read:
        read.remove(-1)
    if len(read)==0:
        return -1

    num_inputs = len(read[0][0])

    inputs_by_index = []
    for i in range (0,len(read[0][0])):
        inputs_by_index+=[[read[j][0][i] for j in range (0,len(read))]]
    # for i in inputs_by_index:
    #     print ([j.shape for j in i])

    concatenated_inputs = [np.concatenate(i,axis=0) for i in inputs_by_index]

    # concatenatedInputs = [np.concatenate([read[i][0][input] for i in range (0, len(read))], axis=0) for input in range (0,num_inputs)]
    return concatenated_inputs, np.concatenate([x[1] for x in read], axis=0)


class GameTrainingWrapper:
    def __init__(self, num_players, action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator, q_epsilon, verbose):
        self.game = game.CoupGame(num_players)

        self.action_evaluator = action_evaluator
        self.assassin_block_evaluator = assassin_block_evaluator
        self.aid_block_evaluator = aid_block_evaluator
        self.captain_block_evaluator = captain_block_evaluator
        self.challenge_evaluator = challenge_evaluator

        self.action_evaluation_data_queues = []
        self.assassin_block_evaluation_data_queues = []
        self.aid_block_evaluation_data_queues = []
        self.captain_block_evaluation_data_queues = []
        self.challenge_evaluation_data_queues = []
        for i in range (0, num_players):
            self.action_evaluation_data_queues += [ActionEvaluatorQueue()]
            self.assassin_block_evaluation_data_queues += [ActionEvaluatorQueue()]
            self.aid_block_evaluation_data_queues += [ActionEvaluatorQueue()]
            self.captain_block_evaluation_data_queues += [ActionEvaluatorQueue()]
            self.challenge_evaluation_data_queues += [ActionEvaluatorQueue()]

        #self.hand_states = np.full((game.MAX_PLAYERS, 5), .4, dtype=np.float32)
        self.all_data_queues = [self.action_evaluation_data_queues, self.assassin_block_evaluation_data_queues, self.aid_block_evaluation_data_queues, self.captain_block_evaluation_data_queues, self.challenge_evaluation_data_queues]

        self.q_epsilon = q_epsilon
        self.verbose = verbose

        self.next_turn_q_biases = [0]*game.MAX_PLAYERS

    def print_game_state(self):
        print ("Raw hands:", self.game.hands)
        print ("Hands: ", [game.cards_to_names(i) for i in self.game.hands])
        print("Coins: ", self.game.player_coins)


    def decide_challenge(self, challenger, challengee, action, write_decision_to_training):

        nondiscarded_cards = self.game.count_inplay()

        prior_probability = rows_to_first_second(np.random.normal(.5, .5, (6, 5)).astype(np.float32), challenger, challengee)

        challenger_cards = self.game.one_hot_hand(challenger)

        num_cards = rows_to_first_second(self.game.hand_sizes(), challenger, challengee)
        num_coins = rows_to_first_second(self.game.player_coins, challenger, challengee)

        noise = np.random.normal(.5, .5, (5,)).astype(np.float32)

        challengable_action = np.zeros((game.NUM_CHALLENGABLE_ACTIONS,))
        challengable_action[game.CHALLENGABLE_ACTIONS.index(action)]=1

        if random() > self.q_epsilon:
            predicted_values = self.challenge_evaluator.predict([
                zero_axis_tile(nondiscarded_cards, 2),
                zero_axis_tile(challenger_cards, 2),
                zero_axis_tile(prior_probability, 2),
                zero_axis_tile(num_cards, 2),
                zero_axis_tile(num_coins, 2),
                zero_axis_tile(noise, 2),
                zero_axis_tile(challengable_action, 2),
                np.array([[0],[1]], dtype=np.float32),
            ]).flatten()
            #decision=np.random.choice(2,1,p=predicted_values/np.sum(predicted_values))[0]
            decision = np.argmax(predicted_values)
        else:
            decision = randrange(2)



        if write_decision_to_training:
            self.challenge_evaluation_data_queues[challenger].append_inputs((
                nondiscarded_cards,
                challenger_cards,
                prior_probability,
                num_cards,
                num_coins,
                noise,
                challengable_action,
                np.array([1 if decision else 0], dtype=np.float32),
            ))

        if self.verbose and decision>0:
            print (challenger, "challenged" + ((" with expected value "+str(np.max(predicted_values))) if 'predicted_values' in vars() else "" ) )
        return (decision, np.max(predicted_values) if 'predicted_values' in vars() else 0)

    def decide_communal_challenge(self, challengers, challengee, action):
        poss_chal = randrange(0, len(challengers))
        result = self.decide_challenge(challengers[poss_chal], challengee, action, write_decision_to_training=True)
        return (result, challengers[poss_chal])

    def decide_block(self, blocker, blockee, action, write_decision_to_training):
        blocker=int(blocker)
        blockee=int(blockee)
        is_captain=False
        if action==game.ASSASSINATE:
            evaluator = self.assassin_block_evaluator
            data_queue = self.assassin_block_evaluation_data_queues[blocker]
        elif action==game.FOREIGN_AID:
            evaluator = self.aid_block_evaluator
            data_queue = self.aid_block_evaluation_data_queues[blocker]
        else:
            is_captain=True
            evaluator = self.captain_block_evaluator
            data_queue = self.captain_block_evaluation_data_queues[blocker]

        nondiscarded_cards = self.game.count_inplay()

        prior_probability = rows_to_first_second(np.random.normal(.5, .5, (6, 5)).astype(np.float32), blocker,
                                                 blockee)

        blocker_cards = self.game.one_hot_hand(blocker)

        num_cards = rows_to_first_second(self.game.hand_sizes(), blocker, blockee)
        num_coins = rows_to_first_second(self.game.player_coins, blocker, blockee)

        noise = np.random.normal(.5, .5, (5,)).astype(np.float32)

        options = 3 if is_captain else 2
        option_array = np.array([[0,0],[1,0],[0,1]], dtype=np.float32) if is_captain else np.array([[0],[1]], dtype=np.float32)

        if random()>self.q_epsilon:
            predicted_values = evaluator.predict([
                zero_axis_tile(nondiscarded_cards, options),
                zero_axis_tile(blocker_cards, options),
                zero_axis_tile(prior_probability, options),
                zero_axis_tile(num_cards, options),
                zero_axis_tile(num_coins, options),
                zero_axis_tile(noise, options),
                option_array,
            ]).flatten()
            #decision_index = np.random.choice(3 if is_captain else 2, 1, p=predicted_values / np.sum(predicted_values))[0]
            decision_index = np.argmax(predicted_values)
        else:
            decision_index = randrange(3 if is_captain else 2)

        decision=decision_index
        if is_captain:
            if decision_index==1:
                decision=game.CAPTAIN
            if decision_index==2:
                decision=game.AMBASSADOR

        if write_decision_to_training:
            data_queue.append_inputs((
                nondiscarded_cards,
                blocker_cards,
                prior_probability,
                num_cards,
                num_coins,
                noise,
                option_array[decision_index],
            ))
        if self.verbose and decision>0:
            print (blocker, "blocked" + ((" with expected value "+str(np.max(predicted_values))) if 'predicted_values' in vars() else "" ) )
        return (decision, np.max(predicted_values) if 'predicted_values' in vars() else 0)

    def decide_communal_block(self, blockers, blockee, action):
        wv=self.verbose
        self.verbose=False
        results = [self.decide_block(x, blockee, action, write_decision_to_training=True) for x in
                   blockers]
        self.verbose=wv
        if (True in [x[0] for x in results]):  # If someone decided to challenge
            max_index = 0
            for i in range(1, len(blockers)):
                if results[i][1] > results[max_index][1] and results[i][0]:
                    max_index = i
            # print ("Communal responder: ", blockers[max_index])
            if self.verbose:
                print(blockers[max_index], "blocked")
            return (True, blockers[max_index])

        else:
            return (False, -1)

    # def replace_card(self, player, card):
    #     self.game.hands[player].remove(card)
    #     self.game.deck+=[card]
    #     self.game.shuffle()
    #     self.game.hands[player]+=[self.game.draw_from_deck()]

    def lose_card(self, player):
        if len(self.game.hands[player])>0:
            self.game.discards+=[self.game.hands[player].pop(0)]

        LOSS_BIAS=.3
        self.next_turn_q_biases[player]-=LOSS_BIAS  # Bias for losing a card
        for i in range (game.MAX_PLAYERS):
            self.next_turn_q_biases[i]+=LOSS_BIAS/self.game.num_players


    def coup(self, actor, target):
        self.game.player_coins[actor] -= 7
        self.lose_card(target)

    def foreign_aid(self, actor):
        self.game.player_coins[actor] += 2

    def income(self, actor):
        self.game.player_coins[actor] += 1

    def tax(self, actor):
        self.game.player_coins[actor] += 3

    def assassinate(self, actor, target):
        self.game.player_coins[actor] -= 3
        self.lose_card(target)

    def steal(self, actor, target):
        stolen = min(self.game.player_coins[target], 2)
        self.game.player_coins[target] -= stolen
        self.game.player_coins[actor] += stolen

    def exchange(self, actor): #Exchanges cards. Currently random.
        self.game.hands[actor]+=[self.game.draw_from_deck(), self.game.draw_from_deck()]
        shuffle(self.game.hands[actor])
        self.game.deck+=[self.game.hands[actor].pop(0), self.game.hands[actor].pop(1)]
        self.game.hands[actor]=self.game.hands[actor][:2]
        self.game.shuffle()




    def take_turn(self):
        turn_taker=self.game.turn

        undiscarded_cards = self.game.count_inplay()
        prior_probability = row_to_first(np.random.normal(.5,.5,(6,5)).astype(np.float32), turn_taker)
        turn_taker_cards = self.game.one_hot_hand(turn_taker)
        num_cards = row_to_first(self.game.hand_sizes(), turn_taker)
        num_coins = row_to_first(self.game.player_coins, turn_taker)
        noise = np.random.normal(.5,.5,(5,)).astype(np.float32)

        targets = self.game.players_in()
        targets.remove(turn_taker)
        rel_targets = [(t-turn_taker-1)%game.MAX_PLAYERS for t in targets]  # Shift targets to align with their input info
        num_targets = len(targets)
        tlist = []
        alist = []
        for a in game.ACTIVE_ACTIONS:
            if (a != game.COUP or num_coins[0] >= 7) and (a == game.COUP or num_coins[0] < 10) and (
                    a != game.ASSASSINATE or num_coins[0] >= 3):  # If the action is in fact valid
                alist += [a] * num_targets
                tlist += rel_targets
        action_inputs = np.array(alist, dtype=np.float32)
        target_inputs = np.array(tlist, dtype=np.float32)

        num_options = action_inputs.shape[0]

        inputs = [
            zero_axis_tile(undiscarded_cards, num_options),
            zero_axis_tile(turn_taker_cards, num_options),
            zero_axis_tile(prior_probability, num_options),
            zero_axis_tile(num_cards, num_options),
            zero_axis_tile(num_coins, num_options),
            zero_axis_tile(noise, num_options),

            one_hot(action_inputs, game.NUM_ACTIVE_ACTIONS),
            one_hot(target_inputs, game.MAX_PLAYERS-1),
        ]
        predicted_rewards = self.action_evaluator.predict(inputs).flatten()

        # Choose the action for our next move.
        if random()>self.q_epsilon:
            choice_index = np.argmax(predicted_rewards)
        else:
            choice_index = randrange(num_options)


        action = action_inputs[choice_index]
        target = target_inputs[choice_index]

        target = (target+1+turn_taker) %game.MAX_PLAYERS


        action = int(action)
        target = int(target)

        # Append the best state this turn to the output stack for any decisions made in the past turn
        for qset in self.all_data_queues:
            q=qset[turn_taker]
            while q.num_inputs()>q.num_outputs():
                q.append_output(np.array([np.max(predicted_rewards) + self.next_turn_q_biases[turn_taker]]))

        # Reset the bias
        self.next_turn_q_biases[turn_taker] = 0

        # Append the inputs which gave our current action to the input stack. Output won't be known until next turn
        self.action_evaluation_data_queues[turn_taker].append_inputs(
            [x[choice_index] for x in inputs]
        )
        if self.verbose:
            print("\n")
            print("Turn:", turn_taker)
            print("Deck:", game.cards_to_names(self.game.deck))
            print("Hands: ", [game.cards_to_names(i) for i in self.game.hands])
            print("Coins: ", self.game.player_coins)
            print("Action:", game.ACTION_REFERENCE[action])
            print("Target:", target)

        if (action == game.COUP):
            self.coup(turn_taker, target)

        if (action == game.INCOME):
            self.income(turn_taker)

        if (action == game.FOREIGN_AID):
            communal_block_results = self.decide_communal_block(targets, turn_taker, game.FOREIGN_AID)
            if communal_block_results[0]: #If one wants to block, pick the maximum  (note: this could cause internal training problems -- "just-below-max-ism"
                blocking_player = communal_block_results[1]

                challenge_info = self.decide_challenge(turn_taker, blocking_player, game.BLOCK_FOREIGN_AID, write_decision_to_training=True)

                if challenge_info[0]: #If the block was challenged
                    if self.game.has_card(blocking_player, game.DUKE):
                        self.lose_card(turn_taker)
                        self.game.replace(blocking_player, game.DUKE)
                    else:
                        self.lose_card(blocking_player)
                        self.foreign_aid(turn_taker)

            else: # If no block, write the decision not to block to the training queue for each player, and call the foreign aid
                self.foreign_aid(turn_taker)

        if (action == game.EXCHANGE):
            communal_challenge_results = self.decide_communal_challenge(targets, turn_taker, action)
            if communal_challenge_results[0]:
                if self.game.has_card(turn_taker, game.AMBASSADOR):
                    self.lose_card(communal_challenge_results[1])
                    self.game.replace(turn_taker, game.AMBASSADOR)
                    self.exchange(turn_taker)
                else:
                    self.lose_card(turn_taker)
            else:
                self.exchange(turn_taker)

        if (action == game.TAX):
            communal_challenge_results = self.decide_communal_challenge(targets, turn_taker, action)
            if communal_challenge_results[0]:
                if self.game.has_card(turn_taker, game.DUKE):
                    self.lose_card(communal_challenge_results[1])
                    self.game.replace(turn_taker, game.DUKE)
                    self.tax(turn_taker)
                else:
                    self.lose_card(turn_taker)
            else:
                self.tax(turn_taker)

        if (action == game.ASSASSINATE):
            blocking_info = self.decide_block(target, turn_taker, game.ASSASSINATE, write_decision_to_training=True)
            if blocking_info[0]: #If they are blocking, we can challenge
                self.game.player_coins[turn_taker] -= 3
                challenge_info = self.decide_challenge(turn_taker, target, game.BLOCK_ASSASSINATE, write_decision_to_training=True)
                if challenge_info[0]:
                    if self.game.has_card(target, game.CONTESSA):
                        self.lose_card(turn_taker)
                        self.game.replace(target, game.CONTESSA)
                    else:
                        self.lose_card(target)
                        self.lose_card(target)

            else:
                challenge_info = self.decide_challenge(turn_taker, target, game.ASSASSINATE, write_decision_to_training=True)
                if challenge_info[0]:
                    if self.game.has_card(turn_taker, game.ASSASSIN):
                        self.lose_card(target)
                        self.assassinate(turn_taker, target)
                        self.game.replace(turn_taker, game.ASSASSIN)
                    else:
                        self.lose_card(turn_taker)
                else:
                    self.assassinate(turn_taker, target)

        if (action == game.STEAL):
            blocking_info = self.decide_block(target, turn_taker, game.STEAL, write_decision_to_training=True)
            if blocking_info[0]: #If they are blocking, we can challenge
                blocking_card = game.CAPTAIN if blocking_info[0]==1 else game.AMBASSADOR
                blocking_action = game.BLOCK_STEAL_CAPTAIN if blocking_info[0]==1 else game.BLOCK_STEAL_AMBASSADOR

                challenge_info = self.decide_challenge(turn_taker, target, blocking_action, write_decision_to_training=True)
                if challenge_info[0]:
                    if self.game.has_card(target, blocking_card):
                        self.lose_card(turn_taker)
                        self.game.replace(target, blocking_card)
                    else:
                        self.lose_card(target)
                        self.steal(turn_taker, target)

            else:
                challenge_info = self.decide_challenge(turn_taker, target, game.STEAL, write_decision_to_training=True)
                if challenge_info[0]:
                    if self.game.has_card(turn_taker, game.CAPTAIN):
                        self.lose_card(target)
                        self.steal(turn_taker, target)
                        self.game.replace(turn_taker, game.CAPTAIN)
                    else:
                        self.lose_card(turn_taker)
                else:
                    self.steal(turn_taker, target)

        players_alive=0
        for i in range (self.game.num_players): # Fill in 0s for rewards for any eliminated players, and set their attributes to 0
            if self.game.hands[i]==[]:
                for queue_type in self.all_data_queues:
                    while queue_type[i].num_outputs()<queue_type[i].num_inputs():
                        queue_type[i].append_output(np.array([0], dtype=np.float32))
                self.game.player_coins[i]=0

            else:
                players_alive+=1


        self.game.next_turn()

        if players_alive>1:
            return True
        else: #If the game is over, fill in 1s for rewards for any surviving players
            for i in range(self.game.num_players):
                if self.game.hands[i] != []:
                    for queue_type in self.all_data_queues:
                        while queue_type[i].num_outputs() < queue_type[i].num_inputs():
                            queue_type[i].append_output(np.array([1], dtype=np.float32))
                    if self.verbose:
                        print ("Player", i, "won")
            return False

    def train_evaluator(self, data_queue_list, evaluator, verbose):
        data=combine_ready_from_list(data_queue_list)
        if data!=-1:
            evaluator.fit(x=data[0], y=data[1], batch_size=32, epochs=1, verbose=verbose)
    def train_all_evaluators(self, verbose=0):
        self.train_evaluator(self.action_evaluation_data_queues, self.action_evaluator, verbose)
        self.train_evaluator(self.assassin_block_evaluation_data_queues, self.assassin_block_evaluator, verbose)
        self.train_evaluator(self.captain_block_evaluation_data_queues, self.captain_block_evaluator, verbose)
        self.train_evaluator(self.aid_block_evaluation_data_queues, self.aid_block_evaluator, verbose)
        self.train_evaluator(self.challenge_evaluation_data_queues, self.challenge_evaluator, verbose)
