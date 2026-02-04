import pybullet as p
import time
import pybullet_data
import os
import sys
import cv2
import numpy as np
import itertools

from pathlib import Path
from IK_solver import inverseKinematic, forwardKinematic
from path_manager import PathManager

#For the vision
ROOT = Path(__file__).resolve().parent.parent
DETECT_SRC = ROOT / "Detect" / "src"

for p in [str(DETECT_SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Detect.src.vision_system import VisionSystem as vs
except ImportError as e:
    print(f"Erreur d'importation (vision system) : {e}")

try:
    from Detect.src.simulation.environment import RobotEnvironment
    from Detect.src.simulation.camera import Camera
except ImportError as e:
    print(f"Erreur d'importation (simulation/camera) : {e}")


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
        #Indices into self.jointStates array (motor joints only)
        legMap = {
            "BL": [0, 1, 2],
            "BR": [3, 4, 5],
            "FR": [6, 7, 8],
            "FL": [9, 10, 11]
        }
        
        if leg not in legMap:
            print(f"ERROR: Invalid leg '{leg}'. Must be FL, FR, BL, or BR")
            return None
        
        return legMap[leg]

    def getLegPosition(self, leg, doPrint=False):
        joints_indices = self.getLegMap(leg)
        if joints_indices is None:
            return None
        
        #Convert self.jointStates indices to PyBullet joint indices
        pybullet_joints = [self.motor_joints[i] for i in joints_indices]
        
        #Get current joint angles
        joint_states = p.getJointStates(self.id, pybullet_joints)
        t1 = joint_states[0][0]
        t2 = joint_states[1][0]
        t3 = joint_states[2][0]
        
        #Calculate forward kinematics
        T03 = forwardKinematic(t1, t2, t3)
        
        #Extract position from transformation matrix
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
    
    def verifyDHParameters(self, leg, doPrint=True):
        joints_indices = self.getLegMap(leg)
        if joints_indices is None:
            return None
        
        pybullet_joints = [self.motor_joints[i] for i in joints_indices]
        
        # Get current joint angles
        joint_states = p.getJointStates(self.id, pybullet_joints)
        t1 = joint_states[0][0]
        t2 = joint_states[1][0]
        t3 = joint_states[2][0]
        
        # Calculate FK position
        T03 = forwardKinematic(t1, t2, t3)
        fk_position = T03[:3, 3]
        
        # Get the shoulder joint's link state (parent of first joint)
        shoulder_joint_idx = pybullet_joints[0]
        foot_link_idx = pybullet_joints[2] + 1
        
        # Get link states - use link frames, not joint frames
        shoulder_link_state = p.getLinkState(self.id, shoulder_joint_idx)
        foot_link_state = p.getLinkState(self.id, foot_link_idx)
        
        shoulder_pos = np.array(shoulder_link_state[4])  # [4] is local frame origin in world
        shoulder_orn = np.array(shoulder_link_state[5])  # [5] is local frame orientation
        foot_pos = np.array(foot_link_state[4])
        
        # Transform foot position to shoulder's local frame
        # 1. Get vector in world frame
        vec_world = foot_pos - shoulder_pos
        
        # 2. Convert to shoulder's local frame using inverse rotation
        rot_matrix = np.array(p.getMatrixFromQuaternion(shoulder_orn)).reshape(3, 3)
        actual_position = rot_matrix.T @ vec_world
        
        # Calculate error
        error = actual_position - fk_position
        error_magnitude = np.linalg.norm(error)
        
        if doPrint:
            print(f"\n=== {leg} DH Parameter Verification ===")
            print(f"Joint angles: t1={np.degrees(t1):.2f}°, t2={np.degrees(t2):.2f}°, t3={np.degrees(t3):.2f}°")
            print(f"FK result:       x={fk_position[0]:.3f}, y={fk_position[1]:.3f}, z={fk_position[2]:.3f}")
            print(f"PyBullet local:  x={actual_position[0]:.3f}, y={actual_position[1]:.3f}, z={actual_position[2]:.3f}")
            print(f"Error:           x={error[0]:.3f}, y={error[1]:.3f}, z={error[2]:.3f}")
            print(f"Error magnitude: {error_magnitude:.3f}m")
            if error_magnitude < 0.01:
                print("✓ DH parameters appear CORRECT")
            else:
                print("✗ DH parameters may be WRONG")
    
        return {
            "fk_position": fk_position,
            "actual_position": actual_position,
            "error": error,
            "error_magnitude": error_magnitude
        }
    
    def moveLegPyBulletIK(self, leg, target_position):
        """
        Use PyBullet's built-in IK
        target_position: [x, y, z] in BODY/BASE frame
        """
        joints_indices = self.getLegMap(leg)
        if joints_indices is None:
            return False
        
        pybullet_joints = [self.motor_joints[i] for i in joints_indices]
        foot_link_idx = pybullet_joints[2] + 1
        
        # Get body/base transform
        base_pos, base_orn = p.getBasePositionAndOrientation(self.id)
        
        # Convert target from body frame to world frame
        base_rot = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        target_world = np.array(base_pos) + base_rot @ np.array(target_position)
        
        # Get current joint positions to use as rest pose
        current_joint_positions = []
        for i in range(len(self.motor_joints)):
            joint_state = p.getJointState(self.id, self.motor_joints[i])
            current_joint_positions.append(joint_state[0])

        # Define REALISTIC joint limits (adjust these based on your robot's URDF)
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        
        for i in range(len(self.motor_joints)):
            # Tighter limits to prevent flipping
            # Adjust these values based on your robot's actual joint limits
            lower_limits.append(-2.8)  # ~-160 degrees
            upper_limits.append(2.8)   # ~160 degrees
            joint_ranges.append(5.6)   # Range
    
        # Call IK with damping to prefer staying close to rest pose
        result = p.calculateInverseKinematics(
            self.id,
            foot_link_idx,
            target_world,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=current_joint_positions,
            jointDamping=[0.1] * len(self.motor_joints),  # ← ADDED: Prefer staying close to rest pose
            maxNumIterations=100,
            residualThreshold=0.001
        )
        
        # Extract the 3 joints for this leg
        result_idx_0 = self.motor_joints.index(pybullet_joints[0])
        result_idx_1 = self.motor_joints.index(pybullet_joints[1])
        result_idx_2 = self.motor_joints.index(pybullet_joints[2])
        
        self.jointStates[joints_indices[0]] = result[result_idx_0]
        self.jointStates[joints_indices[1]] = result[result_idx_1]
        self.jointStates[joints_indices[2]] = result[result_idx_2]
        
        return True
    
    def printLegCoordinates(self, leg):
        """
        Print foot position in BASE frame (for use with PyBullet IK)
        These coordinates are independent of shoulder rotation
        """
        joints_indices = self.getLegMap(leg)
        if joints_indices is None:
            return None
        
        pybullet_joints = [self.motor_joints[i] for i in joints_indices]
        foot_link_idx = pybullet_joints[2] + 1
        
        # Get base (body) and foot transforms
        base_pos, base_orn = p.getBasePositionAndOrientation(self.id)
        foot_state = p.getLinkState(self.id, foot_link_idx)
        
        base_pos = np.array(base_pos)
        base_orn = np.array(base_orn)
        foot_pos = np.array(foot_state[4])
        
        # Transform foot position to base's local frame
        rot_matrix = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        vec_world = foot_pos - base_pos
        local_position = rot_matrix.T @ vec_world
        
        print(f"{leg}: [{local_position[0]:.3f}, {local_position[1]:.3f}, {local_position[2]:.3f}]")
        return local_position
    
    def printLegCoordinatesForIK(self, leg):
        """
        Print foot position in body frame - works with moveLegPyBulletIK
        """
        joints_indices = self.getLegMap(leg)
        if joints_indices is None:
            return None
        
        pybullet_joints = [self.motor_joints[i] for i in joints_indices]
        foot_link_idx = pybullet_joints[2] + 1
        
        # Get body/base frame
        base_pos, base_orn = p.getBasePositionAndOrientation(self.id)
        
        # Get foot position
        foot_state = p.getLinkState(self.id, foot_link_idx)
        foot_pos = np.array(foot_state[4])
        
        # Transform to body local frame
        base_rot = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        vec_world = foot_pos - np.array(base_pos)
        local_position = base_rot.T @ vec_world
        
        print(f"{leg}: [{local_position[0]:.3f}, {local_position[1]:.3f}, {local_position[2]:.3f}]")
        return local_position
    
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
        #time.sleep(time_step) commented out for training. FOr simulation comment back
    
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

class GaitController:
    def __init__(self, robot, length, height, duration):
        self.controlledRobot = robot
        self.gaitLength = length
        self.gaitHeight = height
        self.gaitDuration = duration

        self.phaseFL = 0
        self.phaseBR = 2  
        self.phaseFR = 4
        self.phaseBL = 6

        self.goalFL = [0, 0, 0]
        self.goalBR = [0, 0, 0]
        self.goalFR = [0, 0, 0]
        self.goalBL = [0, 0, 0]

    def getPhase(self, leg):
        """Get current phase for a leg"""
        phases = {
            "FL": self.phaseFL,
            "BR": self.phaseBR,
            "FR": self.phaseFR,
            "BL": self.phaseBL
        }
        return phases[leg]
    
    def updateAllPhases(self):
        """Update all leg phases simultaneously"""
        self.phaseFL = (self.phaseFL + 1) % 8
        self.phaseBR = (self.phaseBR + 1) % 8
        self.phaseFR = (self.phaseFR + 1) % 8
        self.phaseBL = (self.phaseBL + 1) % 8

    def calculateLegPosition(self, progress, startPos, endPos):
        curGoalPos = startPos + progress *(endPos - startPos) #calculates the neccessary position to achieve movement within duration

        # zOffset = self.gaitHeight * np.sin(progress * np.pi)    #parabolic offset to allow "stepping" motion

        # curGoalPos[2] += zOffset

        return curGoalPos
    
    def executeWalk(self, legName, goalPos, duration = 240):
        # Get current position in BODY frame (not shoulder-local frame)
        joints_indices = self.controlledRobot.getLegMap(legName)
        pybullet_joints = [self.controlledRobot.motor_joints[i] for i in joints_indices]
        foot_link_idx = pybullet_joints[2] + 1
        
        base_pos, base_orn = p.getBasePositionAndOrientation(self.controlledRobot.id)
        foot_state = p.getLinkState(self.controlledRobot.id, foot_link_idx)
        foot_pos = np.array(foot_state[4])
        
        # Transform to body local frame
        base_rot = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        vec_world = foot_pos - np.array(base_pos)
        startPos = base_rot.T @ vec_world
        
        # Add offsets to goal (goal is already in body frame)
        offsetFL = np.array([0.153, 0.387, -0.136])
        offsetFR = np.array([0.175, -0.384, -0.142])
        offsetBL = np.array([-0.237, 0.386, -0.139])
        offsetBR = np.array([-0.202, -0.382, -0.143])

        if legName == "FL":
            goalPos = goalPos + offsetFL
        elif legName == "FR":
            goalPos = goalPos + offsetFR
        elif legName == "BL":
            goalPos = goalPos + offsetBL
        elif legName == "BR":
            goalPos = goalPos + offsetBR

        for progress in range(duration):
            percentage = progress / duration
            goal = self.calculateLegPosition(percentage, startPos, goalPos)
            self.controlledRobot.moveLegPyBulletIK(legName, goal)
            yield False
        yield True

    def sequentialWalk(self, speed):
        bkwduGoal = np.array([0.0, 0.32, 0.1])
        fwduGoal = np.array([0.326, 0.32, 0.1])
        fwdGoal = np.array([0.326, 0.32, -0.066])      # Forward swing
        bkwdGoal = np.array([0.0, 0.32, -0.066])  # Backward push
        
        while True:  # Loop forever
            print("bkwdU")
            yield from self.executeWalk("FL", bkwduGoal, speed)
            print("fwdU")
            yield from self.executeWalk("FL", fwduGoal, speed)
            print("fwd")
            yield from self.executeWalk("FL", fwdGoal, speed)
            print("bkwd")
            yield from self.executeWalk("FL", bkwdGoal, speed)
            # print("ntrl")
            # yield from self.executeWalk("FL", ntrlGoal, speed)

def gaitScheduler(gaitController):
    schedule = [
        ("FL"),
        ("BR"),
        ("FR"),
        ("BL")
    ]
    for leg, goal in schedule:
        yield from gaitController.executeWalk(leg, np.array(goal), 240)

def updatePhase(self):
    curPhase = next(self.phases)

    self.phaseFL = curPhase
    self.phaseBR = curPhase + 1
    self.phaseFR = curPhase + 2
    self.phaseBL = curPhase + 3

def crawlGait(gaitController):
    phase_goals = {
        0: np.array([-0.153, -0.067, 0.236]),   # Lift
        1: np.array([0.173, -0.067, 0.236]),    # Swing
        2: np.array([0.173, -0.067, 0.070]),    # Push back
        3: np.array([0.0, -0.067, 0.070]),      # Push back
        4: np.array([-0.153, -0.067, 0.070]),   # Push back
        5: np.array([-0.153, -0.067, 0.070]),   # Push back
        6: np.array([-0.153, -0.067, 0.070]),   # Push back
        7: np.array([-0.153, -0.067, 0.070])    # Push back
    }
    
    # Create a walker (generator) for each leg
    walkers = {
        "FL": None,
        "BR": None,
        "FR": None,
        "BL": None
    }
    
    while True:  # Continuous gait loop
        all_done = True
        
        # Process ALL legs in the same iteration
        for leg in ["FL", "BR", "FR", "BL"]:
            # Start new movement if walker is done or not started
            if walkers[leg] is None:
                phase = gaitController.getPhase(leg)  # Get current phase for this leg
                goal = phase_goals[phase]
                walkers[leg] = gaitController.executeWalk(leg, goal, duration=240)
            
            # Execute ONE step for this leg
            try:
                done = next(walkers[leg])  # Move leg slightly
                if not done:
                    all_done = False  # Still moving
                else:
                    walkers[leg] = None  # Movement complete
            except StopIteration:
                walkers[leg] = None
        
        # When all legs finish, update phases and restart
        if all_done:
            gaitController.updateAllPhases()
        
        yield False  # Return control to main loop


            

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
    
    # ----- Vision -----
    # Setup environment
    env = RobotEnvironment(gui=True)
    env.spawn_random_scene(num_each_type=1)

    camera = Camera()
    vision = vs(camera, model_path="weights/best.pt")

    # Run detection
    result = vision.detect_and_measure()
            
    # Get target coordinates for path planning
    for detection in result['detections']:
        target_x = detection['position'][0]  # X coordinate
        target_y = detection['position'][1]  # Y coordinate
        object_type = detection['class_name']
    
    print(f"{object_type} at [{target_x:.3f}, {target_y:.3f}]")

    # ----- ----- ----- -----

    # ----- Path planning -----
    path_mgr = PathManager(robot) #import functions for path planning
    if object_type =="cube": #look for a cube to plan a path to it
        target = [target_x, target_y, 0]
        path_mgr.plan_path(target) #planning the path with target and obstacle. This will have to be actualised in real time when we're moving in was of moving environement

    # ------ ----- ----- -----

    robot.print_joint_positions()
    
    #Move into default position
    target_positions = natural_positions.copy()
    
    print(f"\nTarget positions (from URDF): {[f'{p:.3f}' for p in target_positions]}")
    
    print(f"\nRunning simulation with PD control")
    print(f"Kp={config['kp']}, Kd={config['kd']}, Max Force={config['maxForce']}")
    print("Close window or Ctrl+C to exit")
    
    #create gaitController
    gaitController = GaitController(robot, 1.2, 0.2, 240)
    walker = None

    #Run simulation with PD control
    try:
        for i in range(100000):
            walkingMode = "gait"
            if walkingMode == "gait":
                if walker is None:
                   walker = crawlGait(gaitController)
                   #walker = gaitScheduler(gaitController)
                try:
                   next(walker)
                except:
                   walker = None

            elif walkingMode == "rad":
                radGoal = [0, 0.0, 0]
                robot.moveLegRad("FL", radGoal)
                robot.moveLegRad("FR", radGoal)
                robot.moveLegRad("BL", radGoal)
                robot.moveLegRad("BR", radGoal)
            elif walkingMode == "IK":
                #goal = np.array([0, 0, 0.1])  #neutral
                #goal = np.array([0.1, 0, 0.radMode1])  #forward
                goal = np.array([-0.1, 0, 0.1])  #backward
                offsetFL = np.array([0.153, 0.387, -0.136])
                offsetFR = np.array([0.175, -0.384, -0.142])
                offsetBL = np.array([-0.237, 0.386, -0.139])
                offsetBR = np.array([-0.202, -0.382, -0.143])

                robot.moveLegPyBulletIK("FL", goal+offsetFL)
                robot.moveLegPyBulletIK("FR", goal+offsetFR)
                robot.moveLegPyBulletIK("BL", goal+offsetBL)
                robot.moveLegPyBulletIK("BR", goal+offsetBR)
            elif walkingMode == "seq":
                if walker is None:
                    walker = gaitController.sequentialWalk(720)
                try:
                    next(walker)
                except:
                    walker = None
            elif walkingMode == "slider":
                if walker is None:
                    walker = sliderControl(robot)
                try:
                    next(walker)
                except:
                    walker = None
                    
            #update steps and advance simulation
            robot.updateJoints(kp=config["kp"], kd=config["kd"], max_force=config["maxForce"])
            sim.step(config["timeStep"])

            # if i % 300 == 0:
            #     robot.printLegCoordinatesForIK("FL")
            #     robot.printLegCoordinatesForIK("FR")
            #     robot.printLegCoordinatesForIK("BL")
            #     robot.printLegCoordinatesForIK("BR")
                # robot.getLegPosition("FL", True) 
                # robot.getLegPosition("FR", True) 
                # robot.getLegPosition("BL", True)      
                # robot.getLegPosition("BR", True) 
                #robot.verifyDHParameters("FL")
                #print(f"result: t1={np.degrees(angles[0]):.2f}° t2={np.degrees(angles[1]):.2f}° t3={np.degrees(angles[2]):.2f}°")
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


def sliderControl(robot):
    """
    Add GUI sliders to control robot legs and print positions for executeWalk
    """
    import pybullet as p
    
    # Create sliders for FL leg
    slider_fl_x = p.addUserDebugParameter("FL_X", -0.5, 0.5, 0.153)
    slider_fl_y = p.addUserDebugParameter("FL_Y", 0.0, 0.6, 0.387)
    slider_fl_z = p.addUserDebugParameter("FL_Z", -0.3, 0.3, -0.136)
    
    # Create sliders for FR leg
    slider_fr_x = p.addUserDebugParameter("FR_X", -0.5, 0.5, 0.175)
    slider_fr_y = p.addUserDebugParameter("FR_Y", -0.6, 0.0, -0.384)
    slider_fr_z = p.addUserDebugParameter("FR_Z", -0.3, 0.3, -0.142)
    
    # Create sliders for BL leg
    slider_bl_x = p.addUserDebugParameter("BL_X", -0.5, 0.5, -0.237)
    slider_bl_y = p.addUserDebugParameter("BL_Y", 0.0, 0.6, 0.386)
    slider_bl_z = p.addUserDebugParameter("BL_Z", -0.3, 0.3, -0.139)
    
    # Create sliders for BR leg
    slider_br_x = p.addUserDebugParameter("BR_X", -0.5, 0.5, -0.202)
    slider_br_y = p.addUserDebugParameter("BR_Y", -0.6, 0.0, -0.382)
    slider_br_z = p.addUserDebugParameter("BR_Z", -0.3, 0.3, -0.143)
    
    # Print button (using a slider as button)
    print_button = p.addUserDebugParameter("Print Positions (>0.5)", 0, 1, 0)
    
    last_print_state = 0
    
    while True:
        # Read slider values
        fl_x = p.readUserDebugParameter(slider_fl_x)
        fl_y = p.readUserDebugParameter(slider_fl_y)
        fl_z = p.readUserDebugParameter(slider_fl_z)
        
        fr_x = p.readUserDebugParameter(slider_fr_x)
        fr_y = p.readUserDebugParameter(slider_fr_y)
        fr_z = p.readUserDebugParameter(slider_fr_z)
        
        bl_x = p.readUserDebugParameter(slider_bl_x)
        bl_y = p.readUserDebugParameter(slider_bl_y)
        bl_z = p.readUserDebugParameter(slider_bl_z)
        
        br_x = p.readUserDebugParameter(slider_br_x)
        br_y = p.readUserDebugParameter(slider_br_y)
        br_z = p.readUserDebugParameter(slider_br_z)
        
        # Move legs to slider positions
        robot.moveLegPyBulletIK("FL", [fl_x, fl_y, fl_z])
        robot.moveLegPyBulletIK("FR", [fr_x, fr_y, fr_z])
        robot.moveLegPyBulletIK("BL", [bl_x, bl_y, bl_z])
        robot.moveLegPyBulletIK("BR", [br_x, br_y, br_z])
        
        # Check print button
        print_state = p.readUserDebugParameter(print_button)
        if print_state > 0.5 and last_print_state <= 0.5:
            print("\n=== Current Leg Positions (for executeWalk) ===")
            print(f"FL_goal = np.array([{fl_x:.3f}, {fl_y:.3f}, {fl_z:.3f}])")
            print(f"FR_goal = np.array([{fr_x:.3f}, {fr_y:.3f}, {fr_z:.3f}])")
            print(f"BL_goal = np.array([{bl_x:.3f}, {bl_y:.3f}, {bl_z:.3f}])")
            print(f"BR_goal = np.array([{br_x:.3f}, {br_y:.3f}, {br_z:.3f}])")
            print("\n=== Relative to offsets (goals without offsets) ===")
            print(f"FL_relative = np.array([{fl_x - 0.153:.3f}, {fl_y - 0.387:.3f}, {fl_z + 0.136:.3f}])")
            print(f"FR_relative = np.array([{fr_x - 0.175:.3f}, {fr_y + 0.384:.3f}, {fr_z + 0.142:.3f}])")
            print(f"BL_relative = np.array([{bl_x + 0.237:.3f}, {bl_y - 0.386:.3f}, {bl_z + 0.139:.3f}])")
            print(f"BR_relative = np.array([{br_x + 0.202:.3f}, {br_y + 0.382:.3f}, {br_z + 0.143:.3f}])")
        
        last_print_state = print_state
        
        yield False  # Return control to main loop