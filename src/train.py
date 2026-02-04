import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import datetime

from simulation import Simulation, Robot 

class QuadrupedEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(QuadrupedEnv, self).__init__()
        self.render_mode = render_mode
        
        self.sim = Simulation(gui=self.render_mode)
        self.plane = self.sim.add_plane()
        
        urdf_path = "../data/Full_robot_urdf/urdf/Full_robot_urdf.urdf"
        self.robot = self.sim.add_robot(urdf_path, start_pos=[0, 0, 0.3])
        
        #action space: what the ML can control(12 joints)
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        
        #observation space: what the ML "sees". Roll, Pitch, Yaw, RollRate, PitchRate, YawRate, JointPos(12), JointVel (12)
        # 3 + 3 + 12 + 12 = 30
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        
        #training parameter
        self.target_speed = 0.4  #m/s
        self.max_steps = 1000
        self.step_counter = 0
        
        # Define a standing pose (legs slightly bent for stability)
        # Format: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, 
        #          RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
        self.standing_pose = np.array([
            0.0, 0.5, -1.0,  # Front Left
            0.0, 0.5, -1.0,  # Front Right
            0.0, 0.5, -1.0,  # Rear Left
            0.0, 0.5, -1.0   # Rear Right
        ])
        
        self.previous_action = np.zeros(12)
        self.previous_pos = np.array([0, 0, 0.3])
        self.initial_yaw = 0.0
        self.initial_x = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.previous_action = np.zeros(12)
        
        # Convert Euler angles to quaternion and reset pose
        orientation_quat = p.getQuaternionFromEuler([0, 0, 0])
        self.robot.reset_pose(position=[0, 0, 0.35], orientation=orientation_quat)
        
        # Set initial standing pose for all joints
        for i, joint_idx in enumerate(self.robot.motor_joints):
            p.resetJointState(self.robot.id, joint_idx, self.standing_pose[i])
        
        # Store initial heading
        _, orn = self.robot.get_pose()
        _, _, self.initial_yaw = p.getEulerFromQuaternion(orn)
        
        p.resetBaseVelocity(self.robot.id, [0, 0, 0], [0, 0, 0])
        
        # Let robot settle for a moment
        for _ in range(50):
            self.robot.set_all_joints_pd_control(self.standing_pose, kp=5.0, kd=1.0, max_force=30)
            self.sim.step()
        
        self.initial_x = self.robot.get_position()[0]

        #return initial observation
        return self._get_obs(), {}

    def _get_obs(self):
        #Get Base Orientation (Roll, Pitch)
        _, orn = self.robot.get_pose()
        r, p_angle, y = p.getEulerFromQuaternion(orn)
        
        #Get Joint States
        joint_pos, joint_vel = self.robot.get_joint_states()
        
        #Get Velocities
        linear_vel, angular_vel = p.getBaseVelocity(self.robot.id)
        
        # Normalize joint positions slightly for better convergence
        return np.concatenate([
            [r, p_angle, y],            # Orientation (3)
            angular_vel,                # Angular Velocity (3)
            joint_pos,                  # Joint Angles (12)
            joint_vel                   # Joint Velocities (12)
        ]).astype(np.float32)

    def step(self, action):
        self.step_counter += 1
        
        # Blend actions for smoothness
        alpha = 0.99  # Changed from 0.98
        blended_action = alpha * self.previous_action + (1 - alpha) * action
        
        # Scale actions around the standing pose (not zero)
        scaled_action = self.standing_pose + (blended_action * 0.3)  # Keep at 0.3

        # Apply Control with stronger PD gains for stability
        for _ in range(24):  # Keep as is
            self.robot.set_all_joints_pd_control(scaled_action, kp=5.0, kd=1.0, max_force=30)
            self.sim.step()
        
        # Get State
        obs = self._get_obs()
        pos, orn = self.robot.get_pose()
        linear_vel, angular_vel = p.getBaseVelocity(self.robot.id)
        r, pitch, yaw = p.getEulerFromQuaternion(orn)
        
        yaw_error = yaw - self.initial_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        
        # === FORWARD-FOCUSED REWARD FUNCTION ===

        # 1. Forward Velocity - MASSIVE EMPHASIS (primary goal)
        forward_vel = linear_vel[0]
        R_forward_vel = 250.0 * np.clip(forward_vel / self.target_speed, 0, 1.5)  # Up to +30 reward

        # 2. Lateral penalty - SEVERE (must stay straight)
        R_lateral = -10.0 * (linear_vel[1] ** 2)  # Heavy penalty for sideways motion

        # 3. Yaw penalty - SEVERE (must not turn)
        R_yaw = -50.0 * (yaw_error ** 2)  # Reduced from -25.0

        # 3b. Yaw rate penalty - stop spinning
        R_yaw_rate = -15.0 * (angular_vel[2] ** 2)  # Penalize Z-axis rotation

        # 4. Height - moderate (need stability but not primary)
        target_height = 0.28
        height_error = abs(pos[2] - target_height)
        R_height = -1.0 * np.clip(height_error / 0.1, 0, 3)

        # 5. Roll/Pitch - moderate (stability)
        R_roll = -0.5 * (r ** 2)
        R_pitch = -0.5 * (pitch ** 2)

        # 6. Energy - very minimal (allow exploration)
        R_energy = -0.001 * np.sum(scaled_action ** 2)

        # 7. Smoothness - minimal
        action_diff = blended_action - self.previous_action
        R_smooth = -2000.0 * np.sum(action_diff ** 2)  # Changed from -500.0

        # 8. Joint velocity - minimal
        _, joint_vel = self.robot.get_joint_states()
        joint_pos, _ = self.robot.get_joint_states()
        joint_pos_dev = np.sum(np.abs(np.array(joint_pos) - self.standing_pose))
        avg_joint_vel = np.mean(np.abs(np.array(joint_vel)))
        
        # Reward large deviations from standing pose (long movements)
        R_long_movements = 20.0 * np.clip(joint_pos_dev / 2.0, 0, 1)  # Changed from 5.0

        # But heavily penalize if done too fast
        R_joint_vel = -200.0 * avg_joint_vel  # Changed from -50.0

        # 9. Alive bonus
        R_alive = 1.0

        # Reward coordinated, slow leg movements
        R_spider_gait = 100.0 * np.exp(-50.0 * avg_joint_vel)  # Changed from 20.0

        # Get hip joints specifically (every 3rd joint: 0, 3, 6, 9)
        hip_indices = [0, 3, 6, 9]
        hip_positions = [joint_pos[i] for i in hip_indices]
        hip_velocities = [joint_vel[i] for i in hip_indices]

        # Reward large hip deviations (long hip movements)
        hip_pos_dev = np.sum(np.abs(np.array(hip_positions)))
        R_long_hip_movements = 200.0 * np.clip(hip_pos_dev / 1.0, 0, 1)  # Changed from 50.0

        # EXTREME penalty for fast hip movement
        avg_hip_vel = np.mean(np.abs(np.array(hip_velocities)))
        R_hip_vel = -2000.0 * avg_hip_vel  # Changed from -500.0

        # Moderate penalty for other joints (allow them to move faster)
        non_hip_velocities = [joint_vel[i] for i in range(12) if i not in hip_indices]
        avg_non_hip_vel = np.mean(np.abs(np.array(non_hip_velocities)))
        R_other_joint_vel = -10.0 * avg_non_hip_vel  # Changed from -50.0

        # Huge reward for slow hip movement specifically
        R_slow_hip = 500.0 * np.exp(-200.0 * avg_hip_vel)  # Changed from 200.0 and -100.0

        # Total reward - forward dominates everything
        reward = (R_forward_vel + R_lateral + R_yaw + R_yaw_rate + R_height + 
                  R_roll + R_pitch + R_energy + R_smooth + 
                  R_long_hip_movements + R_hip_vel + R_other_joint_vel + R_slow_hip + R_alive)
        
        self.previous_action = blended_action

        # === TERMINATION CONDITIONS ===
        terminated = False

        # Fallen (more lenient initially)
        if pos[2] < 0.05:
            terminated = True
            reward = -50.0

        # Flipped over (more lenient)
        elif abs(r) > 1.5 or abs(pitch) > 1.5:
            terminated = True
            reward = -50.0

        # Moved too far sideways (off course)
        elif abs(pos[1]) > 2.0:
            terminated = True
            reward = -50.0

        # Turned too much
        elif abs(yaw_error) > 2.5:  # Changed from 1.5 to 2.5 rad (~143 degrees)
            terminated = True
            reward = -50.0
    
        truncated = self.step_counter >= self.max_steps
    
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.sim.disconnect()


if __name__ == "__main__":
    # Diagnostic complete: Robot is stable with zero actions
    # Now training with reduced action range (±0.1 rad)
    
    # Create Parallel Environments
    # n_envs=4 amount of environments. Equal to CPU kernels
    vec_env = make_vec_env(lambda: QuadrupedEnv(render_mode=False), n_envs=12, vec_env_cls=SubprocVecEnv)

    #define PPO model (standard MLP)
    # Create log directory
    log_dir = "./training_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Train with tensorboard logging
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,    #log directory
        learning_rate=3e-4,         #progress per ML update
        n_steps=2048,               #samples per environment before policy update
        batch_size=64,              #size for gradient descent
        n_epochs=10,                #training loops through data
        gamma=0.99                  #discount factor. How much long time rewards are worth. Higher value, longer planning
    )

    lengthTraining = 5    #minutes
    averageFPS = 1000
    totalTimesteps = lengthTraining * 60 * averageFPS

    print("Starting training...")
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    model.learn(total_timesteps=totalTimesteps, tb_log_name="quadruped_ppo_run1")

    end_time = datetime.datetime.now()
    training_duration = end_time - start_time

    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training duration: {training_duration}")
    print(f"Total seconds: {training_duration.total_seconds():.2f}")
    
    #save model afterwards

    # Save with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./models/quadruped_ppo_{timestamp}"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    print("Training finished and model saved.")

    #Display result
    print("Visualizing result...")
    env = QuadrupedEnv(render_mode=True)
    obs, _ = env.reset()
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(1/240.)  #normal framerate for display
        if terminated or truncated:
            obs, _ = env.reset()
            time.sleep(1.0)