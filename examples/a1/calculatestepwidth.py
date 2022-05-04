from pycito.systems.A1.a1 import A1VirtualBase

a1 = A1VirtualBase()
a1.Finalize()

q0 = a1.standing_pose()

q, _ = a1.standing_pose_ik(q0[:6], q0)

context = a1.multibody.CreateDefaultContext()
a1.multibody.SetPositions(context, q)
feet = a1.get_foot_position_in_world(context)
print(f"Right foot displacement: {feet[0][0] - feet[2][0]}")
print(f"Left foot displacement: {feet[1][0] - feet[3][0]}")