from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from agent_memory import Memory
import numpy as np
import random


class Agent():
    def __init__(self,possible_actions,starting_mem_len,max_mem_len,starting_epsilon,learn_rate, starting_lives = 5, debug = False):
        self.memory = Memory(max_mem_len)
        self.possible_actions = possible_actions
        self.epsilon = starting_epsilon
        self.epsilon_decay = .9/100000
        self.epsilon_min = .05
        self.gamma = .95
        self.learn_rate = learn_rate
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.lives = starting_lives # Esté parametro no funciona para el env Pong
        self.starting_mem_len = starting_mem_len
        self.learns = 0

# Construcción de una red neuronal convolucional para el agente.
    def _build_model(self):
        model = Sequential()
        model.add(Input((84,84,4)))
        model.add(Conv2D(filters = 32,kernel_size = (8,8),strides = 4,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters = 64,kernel_size = (4,4),strides = 2,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters = 64,kernel_size = (3,3),strides = 1,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        model.add(Dense(512,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(len(self.possible_actions), activation = 'linear'))
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialized\n')
        return model

# Generar un número aleatorio
# Si es menor a la formula implementada anterior, se toma un número aleatorio.
    def get_action(self,state):
        # Explora
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions,1)[0]

        # Toma la mejor acción
        a_index = np.argmax(self.model.predict(state))
        return self.possible_actions[a_index]

# Verifica que todo haya sido validado correctamente
# Se ejecuta como segunda verificación
    def _index_valid(self,index):
        if self.memory.done_flags[index-3] or self.memory.done_flags[index-2] or self.memory.done_flags[index-1] or self.memory.done_flags[index]:
            return False
        else:
            return True

# Entrenar el modelo de red neuronal convolucional en grupos pequeños de 32
    def learn(self,debug = False):
        # Queremos que la salida [a] sea R_(t+1) + Qmax_(t+1).
        # Por lo tanto el target 1 debería ser [output[0], R_(t+1) + Qmax_(t+1), output[2]]

        # Primero se necesitan 32 estados validos
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < 32:
            index = np.random.randint(4,len(self.memory.frames) - 1)
            if self._index_valid(index):
                state = [self.memory.frames[index-3], self.memory.frames[index-2], self.memory.frames[index-1], self.memory.frames[index]]
                state = np.moveaxis(state,0,2)/255
                next_state = [self.memory.frames[index-2], self.memory.frames[index-1], self.memory.frames[index], self.memory.frames[index+1]]
                next_state = np.moveaxis(next_state,0,2)/255

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index+1])
                next_done_flags.append(self.memory.done_flags[index+1])


        # Pasa los estados a través del modelo y los proximos estados pasan por el target
        labels = self.model.predict(np.array(states))
        next_state_values = self.model_target.predict(np.array(next_states))
        
        # Definir la salida
        # Queremos que la salida sea[action_taken] to be R_(t+1) + Qmax_(t+1)
        for i in range(32):
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(next_state_values[i])

        # Entrenar el modelo usando los estados y salidas ya generados
        self.model.fit(np.array(states),labels,batch_size = 32, epochs = 1, verbose = 0)

        # Disminuir el epsilon (Resultado de la formula de predicción).
        # Actualiza cuantas veces el agente ha aprendido.
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1
        
        # Cada 10000 episodios, copia la carga del modelo
        if self.learns % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')
