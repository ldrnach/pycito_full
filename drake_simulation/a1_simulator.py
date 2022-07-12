import os

import pycito.utilities as utils
import pycito.systems.A1.a1 as A1

from pydrake.all import DiagramBuilder, Parser, SceneGraph, MultibodyPlant, LogVectorOutput, DrakeVisualizer, ConnectContactResultsToDrakeVisualizer, Simulator, PiecewisePolynomial

URDF = os.path.join("systems","A1","A1_description","urdf","a1_foot_collision.urdf")

class A1DrakeSimulationBuilder():
    def __init__(self, timestep = 0.01):
        path = utils.FindResource(URDF)
        # Setup the diagram
        self.builder = DiagramBuilder()
        self.scene_graph = self.builder.AddSystem(SceneGraph())
        self.plant = self.builder.AddSystem(MultibodyPlant(dt = timestep))
        self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)
        self.model_id = Parser(plant = self.plant).AddModelFromFile(path, 'quad')

    @classmethod
    def createSimulator(cls, timestep, environment, controller):
        """
        Create a simulator object given the timestep, contact environment, and controller
        
        This is the main class method that should be used as follows:
            simulator, simbuilder = A1DrakeSimulationBuilder.createSimulator(timestep, environment, controller)

        From there, the user can get the plant and controller
        """
        this = cls(timestep)
        this._addEnvironment(environment)
        this._addController(controller)
        this._addVisualizer()
        this._compileDiagram()
        simulator = this._createSimulator()
        return simulator, this

    def _addEnvironment(self, environment):
        """Add the contact environment to the diagram"""
        # Add the environment geometry
        self.plant = environment.addEnvironmentToPlant(self.plant)
        self.plant.Finalize()
        assert self.plant.geometry_source_is_registered()
        # Setup the scene graph
        self.builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.plant.get_geometry_query_input_port()
        )
        self.builder.Connect(
            self.plant.get_geometry_poses_output_port(),
            self.scene_graph.get_source_pose_port(self.plant.get_source_id())
        )

    def _addController(self, controller):
        """Add the controller and logger to the diagram"""
        # Add the controller to the system and wire within builder
        controller_sys = self.builder.AddSystem(controller)
        self.builder.Connect(
            controller_sys.GetOutputPort('torques'),
            self.plant.get_actuation_input_torques(self.model_id)
        )
        self.builder.Connect(
            self.plant.get_state_output_port(),
            controller_sys.GetInputPort('state')
        )    
        # Add the controller logger
        self.logger = LogVectorOutput(controller_sys.GetOutputPort('logging'), self.builder)

    def _addVisualizer(self):
        """Add the Drake visualizer to the diagram"""
        DrakeVisualizer().AddToBuilder(self.builder, self.scene_graph)
        ConnectContactResultsToDrakeVisualizer(self.builder, self.plant, self.scene_graph)

    def _compileDiagram(self):
        self.diagram = self.builder.Build()
        self.diagram.set_name('diagram')
        self.diagram_context = self.diagram.CreateDefaultContext()

    def _createSimulator(self):
        return Simulator(self.diagram, self.diagram_context)

    def get_plant(self):
        """Get the multibody plant object"""
        return self.plant
    
    def get_plant_context(self):
        """Get the mutable plant context from the diagram"""
        return self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)

    def get_logs(self):
        """Return the data from the logger"""
        return self.logger.FindLog(self.diagram_context)

class A1SimulationPlotter():
    """
    Plotting tool for a1 simulations
    """
    def __init__(self):
        self.a1 = A1()
        self.a1.Finalize()

    def plot(self, logs, show=True, savename=None):
        """Convert the logs to trajectories and plot"""
        # Grab the data
        t = logs.sample_times()
        nX = self.a1.multibody.num_positions() + self.a1.multibody.num_velocities()
        x, u = logs.data()[:nX, :], logs.data()[nX:, :]
        # Convert to trajectories
        xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
        utraj = PiecewisePolynomial.ZeroOrderHold(t, u)
        # Plot the data
        self.a1.plot_trajectories(xtraj, utraj, show=show, savename=savename)

if __name__ == '__main__':
    print("Hello from a1_simulator")