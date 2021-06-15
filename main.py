# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from rdl_imp import Agent
import gym
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=[3],tau=0.001,
                  env=env,batch_size=64, layer1_size=400,layer2_size=300, n_actions=1)
    score_history=[]
    np.random.seed(0)
    for i in range(1000):
        obs =env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        score_history.append(score)
        print('episode', i, 'score %.2f' % score, '100 game average %.f' % np.mean(score_history[-100:]))
    filename= 'pendulum.png'
