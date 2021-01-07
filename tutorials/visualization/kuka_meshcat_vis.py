"""
Copied from "Rendering Multibody Plant" Drake tutorial
ONLY WORKS IF RUN IN JUPYTER CELL. Need to have the #%% on the first line.
Luke Drnach
November 20, 2020
"""
#%%
#import os
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as mpl


from pydrake.common import FindResourceOrThrow
from pydrake.geometry import DrakeVisualizer
from pydrake.geometry.render import (ClippingRange, DepthRange, DepthRenderCamera, RenderCameraCore, RenderLabel, MakeRenderEngineVtk, RenderEngineVtkParams)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.sensors import CameraInfo, RgbdSensor

# Helper methods
def xyz_rpy_deg(xyz, rpy_deg):
    """Defines a pose"""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi/180), xyz)

reserved_labels = [
    RenderLabel.kDoNotRender,
    RenderLabel.kDontCare,
    RenderLabel.kEmpty,
    RenderLabel.kUnspecified]

# Create Diagram builder with plant & scenegraph
print("Creating Diagram and Plant")
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
# Add Kuka at origin
parser = Parser(plant)
iiwa_file = FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf")
iiwa_1 = parser.AddModelFromFile(iiwa_file, model_name="iiwa_1")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa_1), X_AB = xyz_rpy_deg([0,0,0], [0,0,0]))
# Add second Kuka 
iiwa_2 = parser.AddModelFromFile(iiwa_file, model_name="iiwa_2")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa_2), X_AB=xyz_rpy_deg([0,1,0],[0,0,0]))
# Add the renderer
print("Adding Renderer")
renderer_name = "renderer"
scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
# Add a camera
depth_camera = DepthRenderCamera(
    RenderCameraCore(
        renderer_name,
        CameraInfo(width=640, height=480, fov_y=np.pi/4),
        ClippingRange(0.01, 10.0),
        RigidTransform()),
    DepthRange(0.01, 10.0)
    )
world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
X_WB = xyz_rpy_deg([4,0,0],[-90,0,90])
sensor = RgbdSensor(world_id, X_PB=X_WB, depth_camera=depth_camera)
builder.AddSystem(sensor)
builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
# Add the Drake Visualizer
print("Connecting to MeshCat")
DrakeVisualizer.AddToBuilder(builder, scene_graph)
# Add meshcat visualizer
meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph,zmq_url="new",open_browser=False)
meshcat_vis.vis.jupyter_cell()
# Finalize plant and build diagram
print("Finalizing")
plant.Finalize()
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
# Create simulation
print("Creating Simulator")
simulator = Simulator(diagram)
simulator.Initialize()
meshcat_vis.vis.render_static()
# Set simulator context
simulator.get_mutable_context().SetTime(0.0)
plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
plant.get_actuation_input_port(iiwa_1).FixValue(plant_context, np.zeros((7,1)))
plant.get_actuation_input_port(iiwa_2).FixValue(plant_context, np.zeros((7,1)))
plant.SetPositions(plant_context,iiwa_1, [0.2, 0.4, 0, 0, 0, 0, 0])
# Reset recording
print("Recording Visualization")
meshcat_vis.reset_recording()
# Start recording and simulate
meshcat_vis.start_recording()
simulator.AdvanceTo(1.0)
# publish recording
meshcat_vis.publish_recording()
# Render meshcat
meshcat_vis.vis.render_static()
print("Finished")
# %%
