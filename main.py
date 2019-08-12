import train, models

action_evaluator = models.get_action_evaluator()
assassin_block_evaluator = models.get_block_evaluator(steal=False)
aid_block_evaluator = models.get_block_evaluator(steal=False)
captain_block_evaluator = models.get_block_evaluator(steal=True)
challenge_evaluator = models.get_challenge_evaluator()


for i in range (10000):
    if i%100==0:
        print (i)
    trainer = train.GameTrainingWrapper(5, action_evaluator, assassin_block_evaluator, aid_block_evaluator, captain_block_evaluator, challenge_evaluator)
    game_continuing=True
    while game_continuing:
        game_continuing = trainer.take_turn()
    trainer.train_all_evaluators(verbose=0)

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
