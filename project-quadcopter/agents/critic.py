'''
keras libs
'''
from keras import layers, models, optimizers
from keras import backend as K

import numpy as np
import random

class Critic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.build_model()
        
    def build_model(self):
        states = layers.Input(shape=(self.state_dim,), name='states')
        actions = layers.Input(shape=(self.action_dim,), name='actions')
        # 
        net_states = layers.Dense(units=200, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=100, kernel_regularizer=layers.regularizers.l2(1e-6))(net_states)
        # net_states = layers.Dropout(rate=0.3)(net_states)
        
        net_actions = layers.Dense(units=200, kernel_regularizer=layers.regularizers.l2(1e-6))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dense(units=100, kernel_regularizer=layers.regularizers.l2(1e-6))(net_actions)
        # net_actions = layers.Dropout(rate=0.3)(net_actions)
        
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        
        # more layers?
        # net = layers.Dense(units=64, kernel_regularizer=layers.regularizers.l2(1e-6))(net)
        Q_values = layers.Dense(units=1, kernel_initializer=layers.initializers.RandomUniform(minval=-0.0003, maxval=0.0003), 
                                name='q_values')(net)
        
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # optimizer = optimizers.Adam(lr=1e-3, decay=0.01)
        optimizer = optimizers.Adam(lr=1e-3)
        self.model.compile(optimizer=optimizer, loss='mse')

        action_grad = K.gradients(Q_values, actions)
        
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_grad)
        
