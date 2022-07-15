import os
import numpy as np
from pycito.utilities import FindResource
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, SpatialInertia, UnitInertia, PrismaticJoint, BallRpyJoint, ConnectMeshcatVisualizer, Simulator, ConstantVectorSource

a1_urdf = os.path.join("systems","A1","A1_description","urdf","a1_no_collision.urdf")
a1_urdf = FindResource(a1_urdf)

def make_virtual_joints(plant, model_index):
    # Add virtual joints to represent the floating base
    zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
    # Create virtual, zero-mass
    xlink = plant.AddRigidBody('xlink', model_index, zeroinertia)
    ylink = plant.AddRigidBody('ylink', model_index, zeroinertia)
    zlink = plant.AddRigidBody('zlink', model_index, zeroinertia)
    # Create the translational and rotational joints
    xtrans = PrismaticJoint("xtranslation", plant.world_frame(), xlink.body_frame(), [1., 0., 0.])
    ytrans = PrismaticJoint("ytranslation", xlink.body_frame(), ylink.body_frame(), [0., 1., 0.])
    ztrans = PrismaticJoint("ztranslation", ylink.body_frame(), zlink.body_frame(), [0., 0., 1.])
    rpyrotation = BallRpyJoint("baseorientation", zlink.body_frame(), plant.GetBodyByName('base', model_index).body_frame())
    # Add the joints to the multibody plant
    plant.AddJoint(xtrans)
    plant.AddJoint(ytrans)
    plant.AddJoint(ztrans)
    plant.AddJoint(rpyrotation)
    return plant

def quaternion_velocity_to_rpy(v):
    ori_rate, trans_rate, joint_rate = v[:3], v[3:6], v[6:]
    return np.concatenate([trans_rate, ori_rate, joint_rate], axis=0)

# Setup the plant - with no gravity
builder = DiagramBuilder()
plant, sc = AddMultibodyPlantSceneGraph(builder, time_step = 0.01)
parser = Parser(plant, sc)
a1_quaternion = parser.AddModelFromFile(a1_urdf, 'quaternion_a1')
a1_rpy = parser.AddModelFromFile(a1_urdf, 'rpy_a1')
plant = make_virtual_joints(plant, a1_rpy)
plant.gravity_field().set_gravity_vector([0, 0, 0])
plant.Finalize()

# Create the visualizer
meshcat = ConnectMeshcatVisualizer(builder, sc, zmq_url='new')
# Connect actuation input ports
for model in [a1_quaternion, a1_rpy]:
    na = plant.num_actuators()
    cvs = builder.AddSystem(ConstantVectorSource(np.zeros((na//2,))))
    builder.Connect(cvs.get_output_port(), plant.get_actuation_input_port(model))
# Build the diagram
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# Set the velocity in the subsystem context
rate = np.pi/4

nv = plant.num_velocities(a1_quaternion)
v = np.zeros((nv, ))
v[0] = rate
v[1] = rate
v[2] = rate
v[3] = 1
v[4] = 1
v_rpy = quaternion_velocity_to_rpy(v)

p_rpy = np.zeros((plant.num_positions(a1_rpy,)))
p_rpy[0] = 1

plant.SetPositions(plant_context, a1_rpy, p_rpy)
plant.SetVelocities(plant_context, a1_rpy, v_rpy)
plant.SetVelocities(plant_context, a1_quaternion, v)

# Run the simulation
simulator = Simulator(diagram, diagram_context)
simulator.Initialize()
simulator.set_target_realtime_rate(1.0)

meshcat.reset_recording()
meshcat.start_recording()
simulator.AdvanceTo(3)
meshcat.publish_recording()