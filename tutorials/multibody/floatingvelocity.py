"""
Floating velocity

Tutorial script to examine the nature of floating base velocity in Drake

Luke Drnach
April 28, 2021
"""

#Imports
import numpy as np
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, Simulator
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer


file_path = "systems/block/urdf/free_block.urdf"
# Set up brick simulator
builder = DiagramBuilder()
plant, scene = AddMultibodyPlantSceneGraph(builder, time_step=0)
Parser(plant, scene).AddModelFromFile(file_path)
# Remove gravity
plantGravity = plant.gravity_field().set_gravity_vector([0, 0, 0])
# Finalize the model
plant.Finalize()
# Add meshcat visualizer
meshcat = ConnectMeshcatVisualizer(builder, scene, zmq_url="new", open_browser=False)
meshcat.vis.jupyter_cell()
# Finish the diagram
diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyMutableContextFromRoot(context)
# Set the base position above the ground by 1m
roll = np.pi/2
plant.SetPositions(plant_context, [np.cos(roll/2), 0, np.sin(roll/2), 0, 0, 0, 1])
# Set the velocity
plant.SetVelocities(plant_context,[1, 0, 0, 0, 0, 0])
# Create the simulator
simulator = Simulator(diagram, context)
simulator.Initialize()
meshcat.load()
meshcat.vis.render_static()
simulator.get_mutable_context().SetTime(0.0)
# Simulate and record
meshcat.reset_recording()
meshcat.start_recording()
simulator.AdvanceTo(5.)
meshcat.publish_recording()
meshcat.vis.render_static()



input("View visualization. Press <ENTER> to end")
print("Finished")