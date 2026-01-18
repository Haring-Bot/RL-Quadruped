import numpy as np
from scipy.optimize import minimize

def forwardKinematic(t1, t2, t3):
    theta1 = np.deg2rad(90) + t1
    alpha1 = 0
    a1 = 68.2
    d1 = 0

    theta2 = t2
    alpha2 = np.deg2rad(90)
    a2 = 195
    d2 = 0

    theta3 = -np.deg2rad(90) + t3
    alpha3 = 0
    a3 = 101.9
    d3 = 0

    T01 = transformationMatrix(theta1, alpha1, a1, d1)
    T12 = transformationMatrix(theta2, alpha2, a2, d2)
    T23 = transformationMatrix(theta3, alpha3, a3, d3)

    T03 = T01@T12@T23

    return T03



def inverseKinematic(goalX, goalY, goalZ):

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
    result = minimize(goalFunction, x0=[0,0,0], args=(target,), bounds=bounds, method="L-BFGS-B")

    print(f"result: t1={np.degrees(result.x[0]):.2f}° t2={np.degrees(result.x[1]):.2f}° t3={np.degrees(result.x[2]):.2f}°")




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
