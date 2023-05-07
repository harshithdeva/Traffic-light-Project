from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
import traci
from sumo_rl.util.gen_route import write_route_file
from sumo_rl import SumoEnvironment
import os
import sys

import gymnasium
sys.modules["gym"] = gymnasium
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# from stable_baselines3.common.vec_env import VecMonitor
# from stable_baselines3.common.policies import MlpPolicy


if __name__ == '__main__':

    sumo_env = SumoEnvironment(
        net_file='nets/sadashivnagar/NetworkFile3.net.xml',
        route_file='nets/sadashivnagar/LowTrafficRerouted.rou.xml',
        # out_csv_name='results/a2chighlow_model/a2c_test',
        out_csv_name='results/a2chighlow_fixed/a2c_test',
        single_agent=True,
        use_gui=False,
        num_seconds=10000,
        min_green=5)

    # eval_env = SumoEnvironment(
    #     net_file='nets/sadashivnagar/NetworkFile3.net.xml',
    #     route_file='nets/sadashivnagar/LowTrafficRerouted.rou.xml',
    #     out_csv_name='results/a2clowlow_model/a2c_test',
    #     single_agent=True,
    #     use_gui=False,
    #     num_seconds=10000,
    #     min_green=5)

    env = VecMonitor(DummyVecEnv([lambda: sumo_env]))
    # eval_env = VecMonitor(DummyVecEnv([lambda: eval_env]))

    model = A2C("MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.001)
    model.load('A2C/A2CHighLogs/best_model.zip')

    obs = env.reset()
    prev_wait = 0
    traffic_signal = None
    rewards_model = []

    for i in range(1000):
        actions, _ = model.predict(obs, state=None, deterministic=False)
        obs, reward, done, info = env.step(actions)
        traffic_signal = sumo_env.get_traffic_signal()
        curr_wait = sum(
            traffic_signal[0].get_accumulated_waiting_time_per_lane())
        reward = prev_wait - curr_wait
        prev_wait = curr_wait
        rewards_model.append(reward)

    print('\nTime:', sumo_env.get_accumulated_waiting_time())
    obs = env.reset()
    signal = 0
    prev_wait = 0
    rewards_fixed = []

# Comment from here

    print('Fixed')
    for i in range(1000):
        if i % 10 == 0:
            if signal == 0:
                signal = 1
            elif signal == 1:
                signal = 2
            else:
                signal = 0

        obs, reward, done, info = env.step([signal])
        traffic_signal = sumo_env.get_traffic_signal()
        traffic_signal[0]._diff_waiting_time_reward()
        curr_wait = sum(
            traffic_signal[0].get_accumulated_waiting_time_per_lane())
        reward = prev_wait - curr_wait
        prev_wait = curr_wait
        rewards_fixed.append(reward)
    print('\nTime:', sumo_env.get_accumulated_waiting_time())
    print(sum(rewards_fixed), sum(rewards_model))
