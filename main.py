#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:55:12 2019

@author: koen
"""

import os
import random
import gym
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize

from DQNClasses.ActionGetter import ActionGetter
from DQNClasses.Atari import Atari
from DQNClasses.DQN import DQN
from DQNClasses.ProcessFrame import ProcessFrame
from DQNClasses.ReplayMemory import ReplayMemory
from DQNClasses.TargetNetworkUpdater import TargetNetworkUpdater


import TrainFunction


#ENV_NAME = 'BreakoutDeterministic-v4'
ENV_NAME = 'PongDeterministic-v4'

tf.reset_default_graph()

# Control parameters
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 200000          # Number of frames the agent sees between evaluations
EVAL_STEPS = 10000               # Number of frames for one evaluation
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network. 
                                 # According to Mnih et al. 2015 this is measured in the number of 
                                 # parameter updates (every four actions), however, in the 
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions, 
                                 # before the agent starts learning
MAX_FRAMES = 30000000            # Total number of frames the agent sees 
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an 
                                 # evaluation episode
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output 
                                 # has the shape (1,1,1024) which is split into two streams. Both 
                                 # the advantage stream and value stream have the shape 
                                 # (1,1,512). This is slightly different from the original 
                                 # implementation but tests I did with the environment Pong 
                                 # have shown that this way the score increases more quickly
LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results. 
                                 # Hessel et al. 2017 used 0.0000625
BS = 32                          # Batch size

PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard
RUNID = 'run_1'
os.makedirs(PATH, exist_ok=True)
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n, 
                                                                atari.env.unwrapped.get_action_meanings()))

# main DQN and target DQN networks:
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)   # (★★)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)               # (★★)

init = tf.global_variables_initializer()
saver = tf.train.Saver()    

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

LAYER_IDS = ["conv1", "conv2", "conv3", "conv4", "denseAdvantage", 
             "denseAdvantageBias", "denseValue", "denseValueBias"]

# Scalar summaries for tensorboard: loss, average reward and evaluation score
with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
    EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
    EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

# Histogramm summaries for tensorboard: parameters
with tf.name_scope('Parameters'):
    ALL_PARAM_SUMMARIES = []
    for i, Id in enumerate(LAYER_IDS):
        with tf.name_scope('mainDQN/'):
            MAIN_DQN_KERNEL = tf.summary.histogram(Id, tf.reshape(MAIN_DQN_VARS[i], shape=[-1]))
        ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)

