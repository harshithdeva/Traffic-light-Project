from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
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

    env = SumoEnvironment(
        net_file='nets/sadashivnagar/NetworkFile3.net.xml',
        route_file='nets/sadashivnagar/HighTrafficRerouted.rou.xml',
        out_csv_name='outputs/sadashivnagar/ppohigh/ppo',
        single_agent=True,
        use_gui=False,
        num_seconds=1000,
        min_green=5,
        max_green=60
    )

    eval_env = SumoEnvironment(
        net_file='nets/sadashivnagar/NetworkFile3.net.xml',
        route_file='nets/sadashivnagar/HighTrafficRerouted.rou.xml',
        out_csv_name='outputs/sadashivnagar/ppohigh/ppo_test',
        single_agent=True,
        use_gui=False,
        num_seconds=1000,
        min_green=5,
        max_green=60,
    )

    env = DummyVecEnv([lambda: env])
    eval_env = VecMonitor(DummyVecEnv([lambda: eval_env]))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001
    )
    # model.load('new_model.pkl')
    eval_callback = EvalCallback(
        eval_env, best_model_save_path='./PPOHighLogs/',
        log_path='./logs/', eval_freq=10000,
        deterministic=False
    )
    model.learn(total_timesteps=100000, callback=eval_callback)
    model.save('PPO_model_high.pkl')
