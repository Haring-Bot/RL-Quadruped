import simulation

def getRobotConfig():
    config = {
        #Simulation parameters
        "gravity": -3.7,
        "timeStep": 1./240.,
        
        #PD Control parameters
        "kp": 1,
        "kd": 0.5,
        "maxForce": 50,
        
        #Robot initial pose
        "startPos": [0, 0, 0.5],
        "startOrientation": [0, 0, 0],
        
        #URDF path
        "urdfRelativePath": "../data/Full_robot_urdf/urdf/Full_robot_urdf.urdf"
        
    }
    return config

def getTrainingConfig():
    config = {
        
    }
    return config

def getEnvironmentConfig():
    config = {

    }
    return config

def getObsConfig():
    config = {
        
    }
    return config

def getRewardConfig():
    config = {
        
    }
    return config

def getCommandConfig():
    config = {
        
    }
    return config


def main():
    print("start")
    robot_config = getRobotConfig()
    simulation.startSimulation(robot_config)

if __name__ == "__main__":
    main()