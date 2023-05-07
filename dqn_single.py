from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
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

    write_route_file(
        'nets/single-intersection/single-intersection-gen.rou.xml', 400000, 100000)

    env = SumoEnvironment(
        net_file='nets/sadashivnagar/NetworkFile3.net.xml',
        route_file='nets/sadashivnagar/HighTrafficRerouted.rou.xml',
        out_csv_name='outputs/single-intersection/a2c',
        single_agent=True,
        use_gui=False,
        num_seconds=100000,
        min_green=5)

    env = DummyVecEnv([lambda: env])
    model = DQN("MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.001)
    model.load('logs/sqn.zip/best_model.zip')
    model.learn(total_timesteps=100000)
    model.save('dqn-sadashiv_novec_high.pkl')
