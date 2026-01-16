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
    
    # Create simulation
    sim = Simulation(gui=True, gravity=-3.7)
    
    # Add plane
    plane = sim.add_plane()
    
    # Get the correct path to your URDF file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "../data/Full_robot_urdf/urdf/Full_robot_urdf.urdf")
    
    # Add robot
    robot = sim.add_robot(urdf_path, start_pos=[0, 0, 1], start_orientation=[0, 0, 0])
    
    # Run simulation
    sim.run(num_steps=1000, time_step=1./240.)
    #sim.run_live()

    print("joints", robot.getNumJoints())
    
    # Get final position and orientation
    final_pos, final_orn = robot.get_pose()
    print(final_pos, final_orn)
    
    # Disconnect
    sim.disconnect()


if __name__ == "__main__":
    startSimulation()