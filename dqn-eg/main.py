import agent_model_epsilon_greedy
import game
from collections import deque
import csv
import gym

env_name = 'PongDeterministic-v4'
action_space = [0, 2, 3] # 0 -> noop; 2 -> left; 3 -> right


# Initialize Agent and environment
agent = agent_model_epsilon_greedy.Agent(action_space,replay_memory_len=50000,replay_memory_max_len=750000, init_epsilon = 1, alpha = 0.00025)
env = gym.make(env_name, render_mode='rgb_array')

scores = deque(maxlen = 100)
max_score = -21

# Uncomment following lines for testing
#agent.model.load_weights('epsilon-greedy.hdf5')
#agent.model_target.load_weights('epsilon-greedy.hdf5')
#agent.epsilon = 0.0

env.reset()

with open('epsilon-greedy.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Episode', 'Steps', 'Score', 'Max Score', 'Epsilon'])

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
    print('Epsilon: ' + str(agent.epsilon))

    data = [str(i), str(agent.total_steps - steps), str(score), str(max_score), str(agent.epsilon)] 

    with open('epsilon-greedy.csv', 'a') as f: 
        writer = csv.writer(f)   
        writer.writerow(data)