import gym
import preprocess_frame as ppf
import numpy as np

# Iniciar un nuevo juego
def initialize_new_game(name, env, agent):
  
    # Para que los datos no se sobre escriban, se ejecuta desde 0 un nuevo el env.
    env.reset()
    starting_frame = ppf.resize_frame(env.step(0)[0])

    dummy_action = 0
    dummy_reward = 0
    dummy_done = False
    for i in range(3):
        agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)

def make_env(name, agent):
    env = gym.make(name)
    return env

# Crear y tomar acciones
def take_step(name, env, agent, score, debug):
    # Pasos para un buen funcionamiento del entrenamiento:

    # 1 y 2: Actualizar los tiempos y guardar las cargas
    agent.total_timesteps += 1
    if agent.total_timesteps % 50000 == 0:
      agent.model.save_weights('recent_weights.hdf5') # Archivo de texto donde guarda los datos ejecutados.
      print('\nWeights saved!')

    # 3: Toma una acción
    next_frame, next_frames_reward, next_frame_terminal, info = env.step(agent.memory.actions[-1])
    
    # 4: Consigue el siguiente estado
    next_frame = ppf.resize_frame(next_frame)
    new_state = [agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1], next_frame]
    # Usar np para convertirlo al formato kera's 
    new_state = np.moveaxis(new_state,0,2)/255 
    new_state = np.expand_dims(new_state,0)
    
    # 5: Obtener la nueva acción utilizando el siguiente estado   
    next_action = agent.get_action(new_state)
    
    # 6: Cuando el juego termine, devuelve un puntaje
    if next_frame_terminal:
        agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)
        return (score + next_frames_reward),True

    # 7: Añade lo aprendido a la memoria
    agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

    # 8: Para limpiar en caso de errores, se renderiza.
    if debug:
        env.render()

    # 9: En caso de exito, el agente aprende de la memoria 
    if len(agent.memory.frames) > agent.starting_mem_len:
        agent.learn(debug)

    return (score + next_frames_reward),False

def play_episode(name, env, agent, debug = False):
    initialize_new_game(name, env, agent)
    done = False
    score = 0
    while True:
        score,done = take_step(name,env,agent,score, debug)
        if done:
            break
    return score
