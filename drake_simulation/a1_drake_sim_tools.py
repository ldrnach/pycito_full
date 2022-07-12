import os
import numpy as np

from pydrake.all import DiagramBuilder, Parser, SceneGraph, MultibodyPlant, SpatialInertia, UnitInertia, PrismaticJoint, BallRpyJoint, RigidTransform, CoulombFriction, HalfSpace, LogVectorOutput, DrakeVisualizer, ConnectContactResultsToDrakeVisualizer, Simulator
from pydrake.all import PiecewisePolynomial as pp

from pycito.utilities import FindResource
from pycito.systems.A1.a1 import A1VirtualBase
from drake_simulation.controllers import BasicController

# SIMULATION PARAMETERS
dt = 0.01       # TIMESTEP
target_realtime_rate = 1.0
sim_time = 5.0

make_plots = True


# Helper functions
def add_virtual_joints(plant, model_index):
    zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
    massbox = SpatialInertia(1, np.zeros((3,)), UnitInertia(0.1, 0.1, 0.1))
    boxlink = plant.AddRigidBody('massbox', model_index, massbox)
    # Create virtual, zero-mass
    xlink = plant.AddRigidBody('xlink', model_index, zeroinertia)
    ylink = plant.AddRigidBody('ylink', model_index, zeroinertia)
    zlink = plant.AddRigidBody('zlink', model_index, zeroinertia)
    # Create the translational and rotational joints
    xtrans = PrismaticJoint("xtranslation", boxlink.body_frame(), xlink.body_frame(), [1., 0., 0.])
    ytrans = PrismaticJoint("ytranslation", xlink.body_frame(), ylink.body_frame(),  [0., 1., 0.])
    ztrans = PrismaticJoint("ztranslation", ylink.body_frame(), zlink.body_frame(),  [0., 0., 1.])
    rpyrotation = BallRpyJoint("baseorientation", zlink.body_frame(), plant.GetBodyByName('base').body_frame())
    # Add the joints to the multibody plant
    plant.WeldFrames(plant.world_frame(), boxlink.body_frame())
    plant.AddJoint(xtrans)
    plant.AddJoint(ytrans)
    plant.AddJoint(ztrans)
    plant.AddJoint(rpyrotation)
    return plant

def get_standing_pose():
    a1 = A1VirtualBase()
    a1.Finalize()
    q0 = a1.standing_pose()
    q,_ = a1.standing_pose_ik(q0[:6], q0)
    return q

def make_simulation_plots(logs, plant):
    t = logs.sample_times()
    state = logs.data()[:plant.num_positions() + plant.num_velocities(), :]
    control = logs.data()[plant.num_positions()+plant.num_velociteis():, :]
    a1 = A1VirtualBase()
    a1.Finalize()
    xtraj = pp.FirstOrderHold(t, state)
    utraj = pp.ZeroOrderHold(t, control)
    a1.plot_trajectories(xtraj, utraj)

# Setup the robot model for A1
urdf_file = os.path.join("systems","A1","A1_description","urdf","a1_foot_collision.urdf")
robot_urdf = FindResource(urdf_file)

builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)

a1 = Parser(plant=plant).AddModelFromFile(robot_urdf, 'a1')
# For consistency, add the virtual  joints
plant = add_virtual_joints(plant, a1)


# Add flat ground with friction
X_BG = RigidTransform()
surface_friction = CoulombFriction(static_friction = 1.0, dynamic_friction=1.0)
plant.RegisterCollisionGeometry(
    plant.world_body(), # The body to which this object is registerd
    X_BG,               # The fixed pose of the geometry frame G in the body frame B
    HalfSpace(),        # Defines the geometry of the object
    'ground_collision', # A name
    surface_friction    #Coulomb friction coefficients
)
plant.RegisterVisualGeometry(
    plant.world_body(),
    X_BG,
    HalfSpace(),
    'ground_visual',
    np.array([0.5, 0.5, 0.5, 1.0])  # Color set to completely opaque
)

plant.Finalize()
assert plant.geometry_source_is_registered(), 'Geometry not registered'
print(f"A1 has {plant.num_positions()} configuration variables and {plant.num_velocities()} velocity variables")
# Add a controller (a PD controller in this case)
controller = builder.AddSystem(BasicController(plant, dt))

q_ref = get_standing_pose()
v_ref = np.zeros((plant.num_velocities(),))
controller.SetReference(q_ref, v_ref)

# Setup the SceneGraph
builder.Connect(
    scene_graph.get_query_output_port(), 
    plant.get_geometry_query_input_port()
)
builder.Connect(
    plant.get_geometry_poses_output_port(),
    scene_graph.get_source_pose_port(plant.get_source_id())
)

# Connect the controller
builder.Connect(
    controller.GetOutputPort('torques'),
    plant.get_actuation_input_port(a1)
)
builder.Connect(
    plant.get_state_output_port(),
    controller.GetInputPort('state')
)

# Add logger
logger = LogVectorOutput(controller.GetOutputPort('output_metrics'), builder)

# Setup the visualizer
DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Compile the diagram
diagram = builder.Build()
diagram.set_name('diagram')
diagram_context = diagram.CreateDefaultContext()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(target_realtime_rate)

# Set the initial state:
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

plant.SetPositions(plant_context, q_ref)
plant.SetVelocities(plant_context, v_ref)

# Run the simulator!
simulator.AdvanceTo(sim_time)

if make_plots:
    logs = logger.FindLog(diagram_context)
    make_simulation_plots(logs, plant)