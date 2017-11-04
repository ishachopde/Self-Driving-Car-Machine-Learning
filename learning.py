#!/usr/bin/python3

from model import build_model 
from RLLearner import RLLearner

model_creator=build_model()

learner = RLLearner(model_creator)

print("Start training")
learner.start_training()

