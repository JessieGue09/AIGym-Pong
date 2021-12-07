import the_agent
import environment

# Variable que guarda el env que se va ejecutar 
# TODO: Debe coincidir con el env del archivo main.py
name = 'PongDeterministic-v4'

agent = the_agent.Agent(possible_actions=[0,2,3],starting_mem_len=50,max_mem_len=750000, starting_epsilon = .5, debug = True, learn_rate = .00025)
env = environment.make_env(name,agent)

# Comienza la ejecución del episodio.
environment.play_episode(name, env,agent, debug = True)
env.close()

