import pybullet as p
import time
import pybullet_data
import os
import numpy as np

from IK_solver import inverseKinematic, forwardKinematic


class Plane:
    def __init__(self, urdf_name="plane.urdf"):
        self.id = p.loadURDF(urdf_name)
    
    def get_id(self):
        return self.id


class Robot:
    def __init__(self, urdf_path, start_pos=[0, 0, 1], start_orientation=[0, 0, 0]):
        self.urdf_path = urdf_path
        self.start_pos = start_pos
        self.start_orientation = p.getQuaternionFromEuler(start_orientation)
        self.id = p.loadURDF(urdf_path, self.start_pos, self.start_orientation)
        self.jointStates = np.zeros(12)

        #joint information
        self.num_joints = p.getNumJoints(self.id)
        self.motor_joints = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.id, i)
            if joint_info[2] in [p.JOINT_REVOLUTE]:
                self.motor_joints.append(i)
        
        #Disable default motors
        self.disable_default_motors()
    
    def get_id(self):
        return self.id
    
    def get_position(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        return pos
    
    def get_orientation(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        return orn
    
    def get_pose(self):
        return p.getBasePositionAndOrientation(self.id)
    
    def reset_pose(self, position=None, orientation=None):
        if position is None:
            position = self.start_pos
        if orientation is None:
            orientation = self.start_orientation
        p.resetBasePositionAndOrientation(self.id, position, orientation)

    def getNumJoints(self):
        return p.getNumJoints(self.id)
    
    def moveJoint(self, jointNumber, goalDeg):
        p.setJointMotorControl2(self.id,
                                jointIndex = jointNumber,
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = goalDeg)

    def print_joint_positions(self):
        print("\n=== Current Joint Positions ===")
        for i in range(self.getNumJoints()):
            joint_info = p.getJointInfo(self.id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            joint_state = p.getJointState(self.id, i)
            position = joint_state[0]
            velocity = joint_state[1]
            
            if joint_type in [p.JOINT_REVOLUTE]:
                print(f"Joint {i} ({joint_name}): position = {position:.3f} rad, velocity = {velocity:.3f}")
            else:
                print(f"Joint {i} ({joint_name}): FIXED")
    
    def set_joint_pd_control(self, joint_indices, target_positions, kp=10, kd=1.2, max_force=500):
        for i, joint_idx in enumerate(joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                positionGain=kp,
                velocityGain=kd,
                force=max_force
            )
    
    def set_all_joints_pd_control(self, target_positions, kp=10, kd=1.2, max_force=500):
        if len(target_positions) != len(self.motor_joints):
            print(f"Warning: Expected {len(self.motor_joints)} positions, got {len(target_positions)}")
            return
        
        self.set_joint_pd_control(self.motor_joints, target_positions, kp, kd, max_force)
    
    def get_joint_states(self):
        joint_states = p.getJointStates(self.id, self.motor_joints)
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        return positions, velocities
    
    def disable_default_motors(self): #disable default motors to enable PD control
        for joint_idx in self.motor_joints:
            p.setJointMotorControl2(
                self.id,
                joint_idx,
                p.VELOCITY_CONTROL,
                force=0
            )

    def getLegMap(self, leg):
        legMap = {
            "FL": [9, 10, 11],
            "FR": [6, 7, 8],
            "BL": [0, 1, 2],
            "BR": [3, 4, 5]
        }
        
        if leg not in legMap:
            print(f"ERROR: Invalid leg '{leg}'. Must be FL, FR, BL, or BR")
            return None
        
        return legMap[leg]

    def getLegPosition(self, leg, doPrint=False):
        joints = self.getLegMap(leg)
        if joints is None:
            return None
        
        # Get current joint angles
        joint_states = p.getJointStates(self.id, joints)
        t1 = joint_states[0][0]  # First joint angle
        t2 = joint_states[1][0]  # Second joint angle
        t3 = joint_states[2][0]  # Third joint angle
        
        # Calculate forward kinematics
        T03 = forwardKinematic(t1, t2, t3)
        
        # Extract position from transformation matrix
        position = T03[:3, 3]
        if doPrint:
            print(f"{leg} position: x={position[0]:.2f} y={position[1]:.2f} z={position[2]:.2f}")
        
        return position

    def moveLeg(self, leg, command):
        joints = self.getLegMap(leg)
        if joints is None:
            return False

        angles = inverseKinematic(command[0], command[1], command[2], self.jointStates[joints])
        for i, joint in enumerate(joints):
            self.jointStates[joint] = angles[i]

        return True
    
    def moveLegRad(self, leg, command):
        joints = self.getLegMap(leg)
        if joints is None:
            return False

        for i, joint in enumerate(joints):
            self.jointStates[joint] = command[i]

        return True
                
    def updateJoints(self, kp=10, kd=1.2, max_force=500):
        self.set_all_joints_pd_control(self.jointStates, kp=kp, kd=kd, max_force=max_force)
    
    def getLegLengths(self, leg, doPrint=False):
        leg_data = {}
        joints = self.getLegMap(leg)
        
        # Get the FK position (in leg's local frame)
        joint_states = p.getJointStates(self.id, joints)
        t1 = joint_states[0][0]
        t2 = joint_states[1][0]
        t3 = joint_states[2][0]
        
        # Use FK to get position relative to shoulder
        T03 = forwardKinematic(t1, t2, t3)
        delta = T03[:3, 3]
        distance_3d = np.linalg.norm(delta)
        
        leg_data[leg] = {
            "dx": delta[0],
            "dy": delta[1],
            "dz": delta[2],
            "total": distance_3d
        }
        
        if doPrint:
            print(f"{leg}: dx={delta[0]:.3f}m, dy={delta[1]:.3f}m, dz={delta[2]:.3f}m, total={distance_3d:.3f}m")

        return leg_data
        

class Simulation:
    def __init__(self, gui=True, gravity=-3.7):
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, gravity)
        self.plane = None
        self.robot = None
    
    def add_plane(self):
        self.plane = Plane()
        return self.plane
    
    def add_robot(self, urdf_path, start_pos=[0, 0, 1], start_orientation=[0, 0, 0]):
        self.robot = Robot(urdf_path, start_pos, start_orientation)
        return self.robot
    
    def step(self, time_step=1./240.):
        p.stepSimulation()
        time.sleep(time_step)
    
    def run(self, num_steps=10000, time_step=1./240.):
        print("Running simulation...")
        for i in range(num_steps):
            self.step(time_step)

    def run_live(self):
        p.setRealTimeSimulation(1)
        
        print("Live interaction started.")
        try:
            while True:
                self.robot.moveJoint(1, 10)
        except KeyboardInterrupt:
            self.disconnect()
    
    def disconnect(self):
        p.disconnect()


def startSimulation(config):
    print("starting simulation")
    
    #Create simulation
    sim = Simulation(gui=True, gravity=config["gravity"])
    
    #Add plane
    plane = sim.add_plane()
    
    #Find URDF file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, config["urdfRelativePath"])
    
    #Add robot
    robot_temp = Robot.__new__(Robot)
    robot_temp.urdf_path = urdf_path
    robot_temp.start_pos = config["startPos"]
    robot_temp.start_orientation = p.getQuaternionFromEuler(config["startOrientation"])
    robot_temp.id = p.loadURDF(urdf_path, robot_temp.start_pos, robot_temp.start_orientation)
    
    #Get joint info
    robot_temp.num_joints = p.getNumJoints(robot_temp.id)
    robot_temp.motor_joints = []
    for i in range(robot_temp.num_joints):
        joint_info = p.getJointInfo(robot_temp.id, i)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            robot_temp.motor_joints.append(i)
    
    #Get initial pose from URDF
    print("\n=== Capturing natural initial pose ===")
    time.sleep(0.1)
    natural_positions = []
    for joint_idx in robot_temp.motor_joints:
        joint_state = p.getJointState(robot_temp.id, joint_idx)
        natural_positions.append(joint_state[0])
        joint_info = p.getJointInfo(robot_temp.id, joint_idx)
        joint_name = joint_info[1].decode('utf-8')
        print(f"Joint {joint_idx} ({joint_name}): natural position = {joint_state[0]:.3f} rad")
    
    #Create actual robot and disable default motors
    p.removeBody(robot_temp.id)
    robot = sim.add_robot(urdf_path, start_pos=config["startPos"], start_orientation=config["startOrientation"])
    
    robot.print_joint_positions()
    
    #Move into default position
    target_positions = natural_positions.copy()
    
    print(f"\nTarget positions (from URDF): {[f'{p:.3f}' for p in target_positions]}")
    
    print(f"\nRunning simulation with PD control")
    print(f"Kp={config['kp']}, Kd={config['kd']}, Max Force={config['maxForce']}")
    print("Close window or Ctrl+C to exit")
    
    #Run simulation with PD control
    try:
        for i in range(100000):
            #robot.set_all_joints_pd_control(target_positions, kp=kp, kd=kd, max_force=max_force)
            sim.step(config["timeStep"])
            if i % 24 == 0:
                #goalPos = [-0.6, 0.64, -0.1] #should result in joints [0.75,0,0]
                goalPos = [-0.672, 0.572, 1.57]
                angles = inverseKinematic(goalPos[0], goalPos[1], goalPos[2])
                #robot.moveLeg("FL", goalPos)
                robot.moveLegRad("FL", [0.75, 0, 0])
                robot.updateJoints(kp=config["kp"], kd=config["kd"], max_force=config["maxForce"])

            if i % 300 == 0:
                print()
                print("=====================")
                current_pos, _ = robot.get_joint_states()
                robot.getLegPosition("FL", True)        
                print(f"result: t1={np.degrees(angles[0]):.2f}° t2={np.degrees(angles[1]):.2f}° t3={np.degrees(angles[2]):.2f}°")
                robot.getLegLengths("FL", True)
                #robot.print_joint_positions()     
                #print(robot.jointStates)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
        print(f"\nFinal joint positions: {target_positions}")
    
    robot.print_joint_positions()
    
    print("joints", robot.getNumJoints())
    
    final_pos, final_orn = robot.get_pose()
    print(final_pos, final_orn)
    
    sim.disconnect()


if __name__ == "__main__":
    startSimulation()