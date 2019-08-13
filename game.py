from random import randint, shuffle
import numpy as np


INCOME = 0
FOREIGN_AID = 1
COUP = 2
TAX = 3
ASSASSINATE = 4
STEAL = 5
EXCHANGE = 6

ACTIVE_ACTIONS = (INCOME, FOREIGN_AID, COUP, TAX, ASSASSINATE, STEAL, EXCHANGE)
BLOCKABLE_ACTIONS = (FOREIGN_AID, ASSASSINATE, STEAL)
TARGETING_BLOCKABLE_ACTIONS = (ASSASSINATE, STEAL)

BLOCK_FOREIGN_AID = 7
BLOCK_ASSASSINATE = 8
BLOCK_STEAL_CAPTAIN = 9
BLOCK_STEAL_AMBASSADOR = 10


BLOCKING_ACTIONS = (BLOCK_FOREIGN_AID, BLOCK_ASSASSINATE, BLOCK_STEAL_CAPTAIN, BLOCK_STEAL_AMBASSADOR)

CHALLENGABLE_ACTIONS = tuple(BLOCKING_ACTIONS) + (TAX, STEAL, ASSASSINATE, EXCHANGE)

ACTION_REFERENCE = {
    INCOME: "Income",
    FOREIGN_AID: "Foreign Aid",
    COUP: "Coup",
    TAX: "Tax",
    ASSASSINATE: "Assassinate",
    STEAL: "Steal",
    EXCHANGE: "Exchange",

    BLOCK_FOREIGN_AID: "Block Foreign Aid",
    BLOCK_ASSASSINATE: "Block Assassinate",
    BLOCK_STEAL_CAPTAIN: "Block Steal (Captain)",
    BLOCK_STEAL_AMBASSADOR: "Block Steal (Ambassador)",
}

def actions_to_names(cardset):
    return [ACTION_REFERENCE[i] for i in cardset]



NUM_ACTIVE_ACTIONS = 7 #income, foreign aid, coup, tax, assassinate, steal, exchange
NUM_BLOCKING_ACTIONS = 4 #block foreign aid, block assassination, block steal with captain, block steal with ambassador
NUM_ACTIONS = NUM_ACTIVE_ACTIONS + NUM_BLOCKING_ACTIONS + 5 + 5 #Above , plus losing any card, plus reshuffling any card

NUM_BLOCKABLE_ACTIONS = 3 #foreign aid, assassinate, steal
NUM_CHALLENGABLE_ACTIONS = 8 #tax, assassinate, steal, exchange, block assassin, block foreign aid, block steal with captain, block steal with ambassador,

#Character states are in order Duke, Assassin, Contessa, Captain, Ambassador

DUKE = 0
ASSASSIN = 1
CONTESSA = 2
CAPTAIN = 3
AMBASSADOR = 4

CARD_REFERENCE = {
    DUKE: "Duke",
    ASSASSIN: "Assassin",
    CONTESSA: "Contessa",
    CAPTAIN: "Captain",
    AMBASSADOR: "Ambassador",
}

def cards_to_names(cardset):
    return [CARD_REFERENCE[i] for i in cardset]


MAX_PLAYERS=6

def count_cards(cardset):
    return (np.array([cardset.count(DUKE),
             cardset.count(ASSASSIN),
             cardset.count(CONTESSA),
             cardset.count(CAPTAIN),
             cardset.count(AMBASSADOR)], dtype=np.float32))


class CoupGame:
    def draw_from_deck(self):
        return (self.deck.pop(randint(0,len(self.deck)-1)))

    def __init__(self, num_players):
        self.num_players=num_players
        self.deck=[DUKE]*3+[ASSASSIN]*3+[CONTESSA]*3+[CAPTAIN]*3+[AMBASSADOR]*3
        self.hands=[]
        for i in range (0,num_players):
            self.hands+=[[self.draw_from_deck(),self.draw_from_deck()]]
        while len(self.hands)<MAX_PLAYERS:
            self.hands+=[[]]
        self.discards=[]
        self.turn=0

        self.player_coins=np.array([2]*num_players+[0]*(MAX_PLAYERS-num_players), dtype=np.float32)

    def has_card(self, player, card):
        return card in self.hands[player]

    def count_discards(self):
        return count_cards(self.discards)

    def count_inplay(self):
        return (3- self.count_discards())

    def one_hot_hand(self,player):
        return count_cards(self.hands[player])

    def hand_sizes(self):
        return np.array([len(i) for i in self.hands], dtype=np.float32)

    def next_turn(self):
        self.turn += 1
        self.turn %= MAX_PLAYERS
        while self.hands[self.turn]==[]:
            self.turn += 1
            self.turn %= MAX_PLAYERS

    def players_in(self):
        pin=[]
        for i in range (0,MAX_PLAYERS):
            if self.hands[i]!=[]:
                pin+=[i]
        return pin

    def shuffle(self):
        shuffle(self.deck)

    def replace(self, player, card):
        self.hands[player].remove(card)
        self.deck+=[card]
        self.shuffle()
        self.hands[player]+=[self.draw_from_deck()]
