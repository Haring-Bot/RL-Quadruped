import numpy as np
from scipy.optimize import minimize

l1 = 0.682
l2 = 0.195
l3 = 0.1019

def forwardKinematic(t1, t2, t3):
    theta1 = np.deg2rad(90) + t1
    #theta1 =t1
    alpha1 = 0
    a1 = l1
    d1 =np.deg2rad(90)

    theta2 = t2
    alpha2 = 0
    a2 = l2
    d2 = 0

    theta3 = t3 + np.deg2rad(90)
    #theta3 =t3
    alpha3 = 0
    a3 = l3
    d3 = 0

    T01 = transformationMatrix(theta1, alpha1, a1, d1)
    T12 = transformationMatrix(theta2, alpha2, a2, d2)
    T23 = transformationMatrix(theta3, alpha3, a3, d3)

    T03 = T01@T12@T23

    return T03



def inverseKinematic(goalX, goalY, goalZ, initialGuess=None, doPrint=False, ):

    bound1 = (-0.75, 0.75)
    bound2 = (0, 0.9)
    bound3 = (-0.6, 2.3)
    bounds = [bound1, bound2, bound3]

    def goalFunction(rotation, targetPos):
        translation = forwardKinematic(rotation[0], rotation[1], rotation[2])
        curPos = translation[:3, 3]
        accuracy = np.linalg.norm(curPos - targetPos)
        return accuracy
    
    target = np.array([goalX, goalY, goalZ])
    x0 = initialGuess if initialGuess is not None else [0, 0, 0]
    result = minimize(goalFunction, x0=x0, args=(target,), bounds=bounds, 
                      method="L-BFGS-B", options={'ftol': 1e-9, 'gtol': 1e-9})
    
    if doPrint:
        print(f"result: t1={np.degrees(result.x[0]):.2f}° t2={np.degrees(result.x[1]):.2f}° t3={np.degrees(result.x[2]):.2f}°")

    return result.x



def transformationMatrix(theta, alpha, a, d):
    matrix = np.zeros((4, 4))

    matrix[0,0] = np.cos(theta)
    matrix[0,1] = -np.sin(theta) * np.cos(alpha)
    matrix[0,2] = np.sin(theta) * np.sin(alpha)
    matrix[0,3] = a * np.cos(theta)

    matrix[1,0] = np.sin(theta)
    matrix[1,1] = np.cos(theta) * np.cos(alpha)
    matrix[1,2] = -np.cos(theta) * np.sin(alpha)
    matrix[1,3] = a * np.sin(theta)

    matrix[2,0] = 0
    matrix[2,1] = np.sin(alpha)
    matrix[2,2] = np.cos(alpha)
    matrix[2,3] = d

    matrix[3,0] = 0
    matrix[3,1] = 0
    matrix[3,2] = 0
    matrix[3,3] = 1

    return matrix

def manualFk(theta1, theta2, theta3):
    r = l1 * np.cos(theta2) * l2 + np.cos(theta2 + theta3 + np.pi/2)
    x = np.sin(theta1) * r
    y = np.cos(theta1) * r
    z = np.sin(theta2) * l2 - np.cos(theta2+theta3) * l3
    return x,y,z

def test_fk_ik():
    print("=== Testing FK and IK ===\n")
    
    # Test 1: Round trip from angles
    test_angles = [
        (-0.6, 0.64, -0.1),
        (0.0, 0.5, 1.0),
        (-0.5, 0.3, 1.5),
        (0.3, 0.7, 0.5)
    ]
    
    for t1, t2, t3 in test_angles:
        # Forward kinematics
        fk_result = forwardKinematic(t1, t2, t3)
        position = fk_result[:3, 3]
        
        # Inverse kinematics
        ik_result = inverseKinematic(position[0], position[1], position[2], 
                                      initialGuess=[t1, t2, t3])
        
        # Check if FK of IK result gives same position
        fk_check = forwardKinematic(ik_result[0], ik_result[1], ik_result[2])
        position_check = fk_check[:3, 3]
        
        print(f"Original angles: [{t1:.3f}, {t2:.3f}, {t3:.3f}]")
        print(f"FK position:     [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        print(f"IK angles:       [{ik_result[0]:.3f}, {ik_result[1]:.3f}, {ik_result[2]:.3f}]")
        print(f"FK(IK) position: [{position_check[0]:.4f}, {position_check[1]:.4f}, {position_check[2]:.4f}]")
        print(f"Position error:  {np.linalg.norm(position - position_check):.6f}")
        print()

# Run the test
#test_fk_ik()

t1 = -0.5
t2 = 0.5
t3 = 0
resultDH = forwardKinematic(t1, t2, t3)
resultManual = manualFk(t1, t2, t3)

errorX = resultDH[0,3] - resultManual[0]
errorY = resultDH[1,3] - resultManual[1]
errorZ = resultDH[2,3] - resultManual[2]

print(f"[DH: {resultDH[0,3]:.3f}, {resultDH[1,3]:.3f}, {resultDH[2,3]:.3f}]")
print(f"manual: {resultManual}")
print(f"error: x={errorX:.2f}, y={errorY:.2f}, z={errorZ:.2f}")