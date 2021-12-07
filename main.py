import the_agent
import environment
import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np

# La variable name manda llamar el enviroment en el que se va ejecutar el juego.
name = 'PongDeterministic-v4'

# Se ejecutan 3 posibles acciones para que la AI puede tomar.
# 
agent = the_agent.Agent(possible_actions=[0,2,3],starting_mem_len=50000,max_mem_len=750000,starting_epsilon = 1, learn_rate = .00025)
env = environment.make_env(name,agent)

last_100_avg = [-21]
scores = deque(maxlen = 100)
max_score = -21

# Hacemos reset al enviroment
env.reset()

for i in range(1000000):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = environment.play_episode(name, env, agent, debug = False) #Cambiar el debug a TRUE para comenzar a rederizar.
    scores.append(score)
    if score > max_score:
        max_score = score

# Acciones a mostrar en la terminal sobre la ejecución en proceso.
    print('\nEpisode: ' + str(i)) # Episodio actual
    print('Duration: ' + str(time.time() - timee)) # Tiempo de ejecución
    print('Score: ' + str(score)) # Puntos anotados por la IA
    print('Max Score: ' + str(max_score)) # Puntaje maximo
    print('Epsilon: ' + str(agent.epsilon)) # Resultado de la formula

    if i%100==0 and i!=0:
        last_100_avg.append(sum(scores)/len(scores))
        plt.plot(np.arange(0,i+1,100),last_100_avg)
        plt.show()