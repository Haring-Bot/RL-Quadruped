import numpy as np

# Your leg segment lengths from IK_solver.py
l1 = 0.682  # Shoulder to waist
l2 = 0.195  # Waist to knee
l3 = 0.1019 # Knee to foot

max_reach = l1 + l2 + l3  # 0.9789 m
safe_reach = max_reach * 0.9  # 0.881 m (90% for safety)

print("=== Leg Reach Analysis ===")
print(f"Maximum theoretical reach: {max_reach:.3f}m")
print(f"Safe reach (90%): {safe_reach:.3f}m")
print()

# Check swing goal
swing_goal = np.array([-0.64, 0.61, 1.57])
swing_distance = np.linalg.norm(swing_goal)
print(f"Swing goal: x={swing_goal[0]:.2f}, y={swing_goal[1]:.2f}, z={swing_goal[2]:.2f}")
print(f"Distance from shoulder: {swing_distance:.3f}m")
print(f"Status: {'❌ UNREACHABLE' if swing_distance > safe_reach else '✅ Reachable'}")
print()

# Check push goal (X and Y only, Z varies)
push_x = 0.64
push_y = 0.61
print(f"Push goal: x={push_x:.2f}, y={push_y:.2f}, z=<varies>")

# Check with different Z values
for z in [-0.3, -0.2, -0.1, 0.0, 0.5, 1.0, 1.57]:
    push_goal = np.array([push_x, push_y, z])
    push_distance = np.linalg.norm(push_goal)
    status = '✅' if push_distance <= safe_reach else '❌'
    print(f"  z={z:5.2f}: distance={push_distance:.3f}m {status}")