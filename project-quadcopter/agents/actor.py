from keras import layers, models, optimizers
from keras import backend as K

import numpy as np
import random

class Actor:
    def __init__(self, state_dim, action_dim, action_low, action_high):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = action_high - action_low
        
        self.build_model()
        
    def build_model(self):
        states = layers.Input(shape=(self.state_dim,), name='states')
        # , kernel_regularizer=layers.regularizers.l2(1e-6)
        net = layers.Dense(units=100, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=200, kernel_regularizer=layers.regularizers.l2(1e-6))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        
        # net = layers.Dense(units=20, activation='relu')(net)  
        # range [0, 1]
        raw_actions = layers.Dense(units=self.action_dim,
                                   kernel_initializer=layers.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                   activation='sigmoid',
                                   name='raw_actions')(net)
        
        actions = layers.Lambda(lambda x: (x*self.action_range)+self.action_low,
                                name='actions')(raw_actions)
        
        self.model = models.Model(inputs=states, outputs=actions)
        
        # 从critic中传回来的误差梯度，shape=[batch, action_dim]
        action_grad = layers.Input(shape=(self.action_dim,))
        # 梯度与变量的乘积为loss，其实loss对action求个导之后就把这里的乘积消除了
        # 最后反传的还是action_grad
        loss = K.mean(-action_grad*actions)
        
        # 添加一个正则？
        # loss += K.sum(K.abs(self.model.trainable_weights()))
        
        optimizer = optimizers.Adam(lr=1e-4)
        update_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        # 外面调用的API
        # 输入参数为模型的输入和critic传回的误差
        self.train_fn = K.function(
            inputs=[self.model.input, action_grad, K.learning_phase()],
            outputs=[],
            updates=update_op
        )
