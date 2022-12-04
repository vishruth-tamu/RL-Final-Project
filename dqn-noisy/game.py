import numpy as np
import cv2

def process_frame(frame):
    frame = frame[30:-12,5:-4]
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame


def new_game(env, agent):
    env.reset()
    init_frame = process_frame(env.step(0)[0])
    action = 0
    reward = 0
    done = False
    for i in range(3):
        agent.replay_memory.add_to_memory(init_frame, reward, action, done)

def play(env, agent, score, debug):
    
    agent.total_steps += 1
    if agent.total_steps % 20000 == 0:
      agent.model.save_weights('noisy-001.hdf5')

    next_frame, next_reward, is_done, _, info = env.step(agent.replay_memory.actions[-1])
    
    # Get next state from replay memory
    next_frame = process_frame(next_frame)
    next_state = [agent.replay_memory.frames[-3], agent.replay_memory.frames[-2], agent.replay_memory.frames[-1], next_frame]
    next_state = np.moveaxis(next_state,0,2)/255 
    next_state = np.expand_dims(next_state,0)
    
    # Get next action
    next_action = agent.select_action(next_state)

    if is_done:
        agent.replay_memory.add_to_memory(next_frame, next_reward, next_action, is_done)
        return (score + next_reward),True

    # Add new state, reward and action to replay memory
    agent.replay_memory.add_to_memory(next_frame, next_reward, next_action, is_done)

    # Train model if sufficient memory available
    if len(agent.replay_memory.frames) > agent.starting_mem_len:
        agent.train(debug)

    return (score + next_reward),False

def play_pong(env, agent, debug = False):
    new_game(env, agent)
    done = False
    score = 0
    while True:
        score,done = play(env,agent,score, debug)
        if done:
            break
    return score