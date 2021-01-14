"""
Class containing boilerplate code for meshcat visualization of a single multibody plant

Luke Drnach
January 14, 2021
"""
#TODO: Either break up the visualizer into three parts (configuration, trajectory, simulation) or run 

# Library imports
import numpy as np
from pydrake.geometry import DrakeVisualizer
from pydrake.all import (TrajectorySource, ConstantVectorSource, MultibodyPositionToGeometryPose, ClippingRange, DepthRange, DepthRenderCamera, RenderCameraCore, RenderLabel, MakeRenderEngineVtk, RenderEngineVtkParams)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.sensors import CameraInfo, RgbdSensor

# Project imports
from utilities import FindResource

# Helper methods
def xyz_rpy_deg(xyz, rpy_deg):
    """Defines a pose"""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi/180), xyz)

class Visualizer():
    def __init__(self,file=None):
        self.file = file
        self._create_diagram()
        self._add_renderer()

    def _create_diagram(self):
        print("Creating diagram")
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, 0.0)
        self.index = Parser(self.plant).AddModelFromFile(self.file)

    def _add_renderer(self):
        print("Adding renderer")
        # Add the renderer
        renderer_name = "renderer"
        self.scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
        # Add a camera
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer_name,
                CameraInfo(width=640, height=480, fov_y=np.pi/4),
                ClippingRange(0.01, 10.0),
                RigidTransform()),
            DepthRange(0.01, 10.0)
        )
        world_id = self.plant.GetBodyFrameIdOrThrow(self.plant.world_body().index())
        X_WB = xyz_rpy_deg([4,0,0],[-90,0,90])
        sensor = RgbdSensor(world_id, X_PB=X_WB, depth_camera=depth_camera)
        # Add the sensor to the diagram
        self.builder.AddSystem(sensor)
        self.builder.Connect(self.scene_graph.get_query_output_port(), sensor.query_object_input_port())

    def _add_meshcat_visualizer(self):
        print("Connecting to MeshCat")
        DrakeVisualizer.AddToBuilder(self.builder, self.scene_graph)
        # Add meshcat visualizer
        self.meshcat_vis = ConnectMeshcatVisualizer(self.builder,self.scene_graph, zmq_url="new",open_browser=False)
        self.meshcat_vis.vis.jupyter_cell()

    def Finalize(self):
        print("Finalizing")
        self.plant.Finalize()
        self.diagram = self.builder.Build()

    def visualize_trajectory(self, xtraj):
        #TODO: Integrate with visualization
        #TODO: Check xtraj is a trajectory
        source = self.builder.AddSystem(TrajectorySource(xtraj))
        pos2pose = self.builder.AddSystem(MultibodyPositionToGeometryPose(self.plant, input_multibody_state=True))
        self.builder.connect(source.get_output_port(), self.scene_graph.get_source_pose_port(self.plant.get_source_id()))
        self.builder.connect(pos2pose.get_output_port(), self.scene_graph.get_source_pose_port(self.plant.get_source_id()))
        simulator = Simulator(self.builder.Build())
        simulator.Initialize()
        simulator.set_target_realtime_rate(1.0)
        self._visualize(xtraj.end_time())

    def visualize_static_configuration(self, q):
        #TODO: Integrate with visualization
        #TODO: Check a is a vector
        source = self.builder.AddSystem(ConstantVectorSource(q))
        pos2pose = self.builder.AddSystem(MultibodyPositionToGeometryPose(self.plant, input_multibody_state=False))
        self.builder.connect(source.get_output_port(), self.scene_graph.get_source_pose_port(self.plant.get_source_id()))
        self.builder.connect(pos2pose.get_output_port(), self.scene_graph.get_source_pose_port(self.plant.get_source_id()))
        simulator = Simulator(self.builder.Build())
        simulator.Initialize()
        simulator.set_target_realtime_rate(1.0)
        self._visualize(1.0)

    def visualize_configuration(self, q):
        simulator = self._create_simulator()
        # Set plant context
        plant_context = self.plant.GetMyContextFromRoot(simulator.get_mutable_context())
        self.plant.get_actuation_input_port().FixValue(plant_context, np.zeros((self.plant.num_actuators(),1)))
        self.plant.SetPositions(plant_context, q)
        # Visualize
        self._visualize(simulator, t=1.0)

    def visualize_simulation(self, x0, utraj):
        simulator = self._create_simulator()
        # Set the plant context

        pass

    def _create_simulator(self):
        simulator = Simulator(self.diagram)
        simulator.Initialize()
        self.meshcat_vis.vis.render_static()
        # Set simulator context
        simulator.get_mutable_context().SetTime(0.0)
        return simulator

    def _visualize(self, simulator, t=1.0):
        print("Recording Visualization")
        self.meshcat_vis.reset_recording()
        # Start recording and simulate
        self.meshcat_vis.start_recording()
        simulator.AdvanceTo(1.0)
        # Publish Recording
        self.meshcat_vis.publish_recording()
        # Render meshcat
        self.meshcat_vis.vis.render_static()
        # Wait for user input to continue
        input("Check visualization in browser. Press <ENTER> when finished >")
        print("Finished")