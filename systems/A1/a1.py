"""
Description of A1 robot

Luke Drnach
November 5, 2020
"""
import numpy as np
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from utilities import FindResource
from pydrake.all import MultibodyPlant, DiagramBuilder, AddMultibodyPlantSceneGraph, SceneGraph, MultibodyPositionToGeometryPose, DrakeVisualizer, Simulator, TrajectorySource, PiecewisePolynomial
from pydrake.geometry.render import DepthCameraProperties, RenderLabel, MakeRenderEngineVtk, RenderEngineVtkParams
from pydrake.multibody.parsing import Parser
from pydrake.systems.sensors import RgbdSensor
from pydrake.math import RigidTransform, RollPitchYaw

def create_a1_multibody():
    file = "systems/A1/A1_description/urdf/a1.urdf"
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.001)
    a1 = Parser(plant).AddModelFromFile(FindResource(file))
    plant.Finalize()
    return(plant, scene_graph, a1)

def create_a1_timestepping():
    pass

def visualize_a1(x):
    file = "systems/A1/A1_description/urdf/a1.urdf"
    plant = MultibodyPlant(0.0)
    scene_graph = SceneGraph()
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    Parser(plant).AddModelFromFile(FindResource(file))
    plant.Finalize()
    builder = DiagramBuilder()
    
    # Create a source trajectory from the input
    if x.shape[0] == plant.num_positions():
        # Append velocities if not given
        v = np.zeros((plant.num_velocities(),))
        x = np.concatenate((x,v), axis=0)

    # Make a trajectory out of the state
    x_traj = PiecewisePolynomial.FirstOrderHold([0., 1.0], np.column_stack((x,x)))    
    # Visualization setup
    builder.AddSystem(scene_graph)
    source = builder.AddSystem(TrajectorySource(x_traj))
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant, input_multibody_state=True))
    builder.Connect(source.get_output_port(0), to_pose.get_input_port())
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))
    visualizer = DrakeVisualizer()
    visualizer.AddToBuilder(builder, scene_graph)
    # build diagram and make visualization
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(x_traj.end_time())
    # Need to actually produce the visualization of A1...

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose"""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg*np.pi/180), xyz)

def a1_meshcat_visualizer(x):
    #Create diagram builder with plant and scene graph
    print("Creating diagram")
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    Parser(plant).AddModelFromFile(FindResource("systems/A1/A1_description/urdf/a1.urdf"))
    # Add the renderer
    print("adding renderer")
    renderer_name = "renderer"
    scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
    # Add a camera - properties are chosen arbitrarily here
    depth_prop = DepthCameraProperties(width=640, height=480, fov_y=np.pi/4, renderer_name=renderer_name, z_near=0.01, z_far=10.)
    world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    X_WB = xyz_rpy_deg([4,0,0],[-90,0,90])
    sensor = RgbdSensor(world_id, X_PB=X_WB, color_properties=depth_prop, depth_properties=depth_prop)
    builder.AddSystem(sensor)
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    # Add Drake Visualizer
    print("adding visualizer")
    DrakeVisualizer.AddToBuilder(builder, scene_graph)
    # Add and show meshcat visualizer
    meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url="new", open_browser=False)
    #meshcat_vis.vis.jupyter_cell()
    # Finalize
    print("finalizing diagram")
    plant.Finalize()
    diagram = builder.Build()
    # Create context
    diagram_context = diagram.CreateDefaultContext()
    sensor_context = sensor.GetMyMutableContextFromRoot(diagram_context)
    sg_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)
    # Publish visualization message
    print("initializing simulator")
    simulator = Simulator(diagram)
    simulator.Initialize()
    # Remote workflow
    print("setting the context")
    meshcat_vis.vis.render_static()
    # Set the context
    simulator.get_mutable_context().SetTime(0.0)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    plant.get_acutation_input_port().FixValue(plant_context, np.zeros((12,1)))
    plant.SetPositions(plant_context,x)
    # Reset the recording
    print("recording the visualization")
    meshcat_vis.reset_recording()
    # Start recording and simulate
    meshcat_vis.start_recording()
    simulator.AdvanceTo(1.0)
    # Publish recording to meshcat
    meshcat_vis.publish_recording()
    # Render meshcat
    meshcat_vis.vis.render_static()


if __name__ == "__main__":
    plant, _, _ = create_a1_multibody()
    print(f"A1 has {plant.num_positions()} position variables and {plant.num_velocities()} velocity variables")
    print(f"A1 has {plant.num_actuators()} actuators")
    print(f"A1 has {plant.num_collision_geometries()} collision geometries")
    context = plant.CreateDefaultContext()
    x = plant.GetPositions(context)
    print(f"Visualizing A1 default configuration")
    a1_meshcat_visualizer(x)