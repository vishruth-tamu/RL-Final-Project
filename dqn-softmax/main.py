import agent_model_softmax
import game
from collections import deque
import csv
import gym

env_name = 'PongDeterministic-v4'
action_space = [0, 2, 3] # 0 -> noop; 2 -> left; 3 -> right

# Initialize Agent and environment
agent = agent_model_softmax.Agent(action_space,replay_memory_len=50000,replay_memory_max_len=750000, init_temp = 0.2, alpha = 0.00025)
env = gym.make(env_name, render_mode='rgb_array')

scores = deque(maxlen = 100)
max_score = -21

# Uncomment following lines for testing
#agent.model.load_weights('softmax_with_temp_decay.hdf5')
#agent.model_target.load_weights('softmax_with_temp_decay.hdf5')

env.reset()

with open('softmax_with_temp_decay.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Episode', 'Steps', 'Score', 'Max Score'])

for i in range(1000):
    
    steps = agent.total_steps
    score = game.play_pong(env, agent, debug = False)
    scores.append(score)
    if score > max_score:
        max_score = score

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_steps - steps))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))

    data = [str(i), str(agent.total_steps - steps), str(score), str(max_score)] 

    with open('softmax_with_temp_decay.csv', 'a') as f: 
        writer = csv.writer(f)   
        writer.writerow(data)