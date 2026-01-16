import pybullet as p
import time
import pybullet_data
import os


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


def startSimulation():
    print("starting simulation")
    
    #Create simulation
    sim = Simulation(gui=True, gravity=-3.7)
    
    #Add plane
    plane = sim.add_plane()
    
    #Find URDF file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "../data/Full_robot_urdf/urdf/Full_robot_urdf.urdf")
    
    #Add robot
    robot_temp = Robot.__new__(Robot)
    robot_temp.urdf_path = urdf_path
    robot_temp.start_pos = [0, 0, 0.5]
    robot_temp.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
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
    robot = sim.add_robot(urdf_path, start_pos=[0, 0, 0.5], start_orientation=[0, 0, 0])
    
    robot.print_joint_positions()
    
    #Move into default position
    target_positions = natural_positions.copy()
    
    print(f"\nTarget positions (from URDF): {[f'{p:.3f}' for p in target_positions]}")
    
    #PD control parameters
    kp = 1
    kd = 0.5
    max_force = 50
    
    print(f"\nRunning simulation with PD control")
    print(f"Kp={kp}, Kd={kd}, Max Force={max_force}")
    print("Close window or Ctrl+C to exit")
    
    #Run simulation with PD control
    try:
        for i in range(100000):
            robot.set_all_joints_pd_control(target_positions, kp=kp, kd=kd, max_force=max_force)
            sim.step()
            
            if i % 1000 == 0:
                current_pos, _ = robot.get_joint_states()
                print(f"Step {i}: Positions = {[f'{p:.2f}' for p in current_pos]}")
    
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