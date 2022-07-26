import os, copy
import numpy as np
import matplotlib.pyplot as plt

import pycito.utilities as utils
from pycito.systems.A1.a1 import A1, A1VirtualBase
import pycito.decorators as deco

from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.geometry import  Role, StartMeshcat
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.multibody.parsing import Parser

from pydrake.all import MeshcatVisualizer, SceneGraph, MultibodyPlant, LogVectorOutput, PiecewisePolynomial, ContactModel, LeafSystem, AbstractValue, ContactResults

URDF = os.path.join("systems","A1","A1_description","urdf","a1_foot_collision.urdf")

#TODO: Refactor. Use https://github.com/RobotLocomotion/drake/blob/master/tutorials/authoring_multibody_simulation.ipynb as a reference
#TODO: Double check conversion from quaternion model to virutal link model

#TODO: Why does the contact logger drop some contact results for some contact points? This is likely an issue with DRAKE

class ContactResultsLogger(LeafSystem):
    def __init__(self, publish_period_seconds: float):
        super().__init__()
        self.DeclarePeriodicPublish(publish_period_seconds)
        self.contact_results_input_port = self.DeclareAbstractInputPort('contact_results',AbstractValue.Make(ContactResults()))
        self.sample_times = []
        self.values = []

    def DoPublish(self, context, event):
        super().DoPublish(context, event)
        self.sample_times.append(context.get_time())
        self.values.append((copy.deepcopy(self.contact_results_input_port.Eval(context))))

    def to_dictionary(self):
        datadict  ={'time': self.sample_times} 
        nT = len(self.sample_times)
        blank = {'force': np.full((3, nT), np.nan),
                'contact_point': np.full((3, nT), np.nan),
                'slip_speed': np.full((nT,), np.nan),
                'penetration_depth': np.full((nT, ), np.nan),
                'contact_normal': np.full((3, nT), np.nan),
                'separation_speed': np.full((nT,), np.nan)}
        for k, log in enumerate(self.values):
            for n in range(log.num_point_pair_contacts()):
                info = log.point_pair_contact_info(n)
                name = log.plant().get_body(info.bodyA_index()).name()
                if name not in datadict:
                    datadict[name] = copy.deepcopy(blank)
                datadict[name]['force'][:, k] = info.contact_force()
                datadict[name]['separation_speed'][k] = info.separation_speed()
                datadict[name]['slip_speed'][k] = info.slip_speed()
                datadict[name]['contact_point'][:,k] = info.contact_point()
                datadict[name]['penetration_depth'][k] = info.point_pair().depth
                datadict[name]['contact_normal'][:,k] = info.point_pair().nhat_BA_W
        return datadict

class A1DrakeSimulationBuilder():
    def __init__(self, timestep = 0.01):
        path = utils.FindResource(URDF)
        # Setup the diagram
        self.timestep = timestep
        self.builder = DiagramBuilder()
        self.scene_graph = self.builder.AddSystem(SceneGraph())
        self.plant = self.builder.AddSystem(MultibodyPlant(time_step = timestep))
        self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)
        self.model_id = Parser(plant = self.plant).AddModelFromFile(path, 'quad')
        self.plant.set_contact_model(ContactModel.kPoint)

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

    @staticmethod
    def get_a1_standing_state():
        """Create an initial state where A1 is standing"""
        a1 = A1()
        a1.Finalize()
        q0 = a1.standing_pose()
        q, status = a1.standing_pose_ik(q0[:7], guess=q0)
        if not status:
            print('IK Failed')
            return False
        else:
            v = np.zeros((a1.multibody.num_velocities(),))
            return np.concatenate([q, v], axis=0)      

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

    def _addController(self, controller_cls):
        """Add the controller and logger to the diagram"""
        # Add the controller to the system and wire within builder
        self.controller = controller_cls(self.plant, self.timestep)
        controller_sys = self.builder.AddSystem(self.controller)
        self.builder.Connect(
            controller_sys.GetOutputPort('torques'),
            self.plant.get_actuation_input_port(self.model_id)
        )
        self.builder.Connect(
            self.plant.get_state_output_port(),
            controller_sys.GetInputPort('state')
        )    
        # Add the controller logger
        self.logger = LogVectorOutput(controller_sys.GetOutputPort('logging'), self.builder)
        # Add the contact results logger
        self.contact_logger = ContactResultsLogger(self.timestep)
        self.builder.AddSystem(self.contact_logger)
        self.builder.Connect(
            self.plant.get_contact_results_output_port(),
            self.contact_logger.GetInputPort('contact_results')
        )

    def _addVisualizer(self):
        """Add the Meshcat visualizer to the diagram"""
        self.meshcat = StartMeshcat()
        self.visualizer = ConnectMeshcatVisualizer(
            self.builder, 
            self.scene_graph, 
            zmq_url='new',
            open_browser=True,
            role = Role.kIllustration,
            prefix = 'visual'
            )
        # self.collision_viz = ConnectMeshcatVisualizer(
        #     self.builder,
        #     self.scene_graph,
        #     zmq_url='default',
        #     open_browser=False,
        #     role = Role.kProximity,
        #     prefix = 'collision'
        # )
        # self.meshcat.SetProperty('collision','visible',False)

    def _compileDiagram(self):
        self.diagram = self.builder.Build()
        self.diagram.set_name('diagram')
        self.diagram_context = self.diagram.CreateDefaultContext()

    def _createSimulator(self):
        """Create and return the simulator object"""
        self.simulator = Simulator(self.diagram, self.diagram_context)
        self.simulator.set_target_realtime_rate(1.0)
        return self.simulator

    def set_initial_state(self, x0):
        """Set the initial state of the plant"""
        context = self.get_plant_context()
        self.plant.SetPositionsAndVelocities(context, x0)

    def initialize_sim(self, target_rate = 1.0):
        """Initialize the simulation"""
        self.simulator.Initialize()
        self.simulator.set_target_realtime_rate(target_rate)

    def run_simulation(self, end_time = 1.0):
        """Run the simulation"""
        self.visualizer.reset_recording()
        self.visualizer.start_recording()
        try:
            self.simulator.AdvanceTo(end_time)
        except:
            print('Simulation failed')
        self.visualizer.publish_recording()

    def get_plant(self):
        """Get the multibody plant object"""
        return self.plant
    
    def get_plant_context(self):
        """Get the mutable plant context from the diagram"""
        return self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)

    def get_logs(self):
        """Return the data from the logger"""
        return self.logger.FindLog(self.diagram_context), self.contact_logger.to_dictionary()

    def get_visualizer(self):
        return self.visualizer

    def get_simulation_data(self):
        """Return the simulation data from the logger as a dictionary"""
        logs = self.logger.FindLog(self.diagram_context)
        nX = self.plant.num_positions() - 1 + self.plant.num_velocities()
        return {'time': np.array(logs.sample_times()),
                'state': logs.data()[:nX, :],
                'control': logs.data()[nX:, :],
        }
class A1SimulationPlotter():
    """
    Plotting tool for a1 simulations
    """
    def __init__(self):
        self.a1 = A1VirtualBase()
        self.a1.Finalize()

    def plot(self, logs, show=True, savename=None):
        """Convert the logs to trajectories and plot"""
        logs, contactlogs = logs[0], logs[1]
        # Grab the data
        t = logs.sample_times()
        nX = self.a1.multibody.num_positions() + self.a1.multibody.num_velocities()
        x, u = logs.data()[:nX, :], logs.data()[nX:, :]
        # Convert to trajectories
        xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
        utraj = PiecewisePolynomial.ZeroOrderHold(t, u)
        # Plot the data
        self.a1.plot_trajectories(xtraj, utraj, show=False, savename=savename)
        # Plot the contactlogs
        self._plot_contact_forces(contactlogs, show=False, savename=utils.append_filename(savename, 'forces'))
        self._plot_contact_results(contactlogs, show=show, savename=utils.append_filename(savename, 'contactdata'))

    @deco.showable_fig
    @deco.saveable_fig
    def _plot_contact_forces(self, contactlogs):
        """
        Plot the contact data from the simulation
        
        Plot:
            x, y, z contact reaction forces
        """
        fig, axs = plt.subplots(3, 1)
        time = np.asarray(contactlogs['time'])
        for key, value in contactlogs.items():
            if key != 'time':
                force = -1*value['force']
                for k in range(3):
                    axs[k].plot(time, force[k,:], linewidth=1.5, label=key)
        axs[0].set_ylabel('Force X (N)')
        axs[1].set_ylabel('Force Y (N)')
        axs[2].set_ylabel('Force Z (N)')
        axs[2].set_xlabel('Time (s)')
        axs[0].legend(frameon=False)
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def _plot_contact_results(self, contactlogs):
        """"
        Plot the contact data from the simulation
        
        Plots:
            contact separation
            contact separation velocity
            contact slip velocity
        """
        fig, axs = plt.subplots(3,1)
        time = np.asarray(contactlogs['time'])
        for key, value in contactlogs.items():
            if key != 'time':
                axs[0].plot(time, value['penetration_depth'], linewidth=1.5, label=key)
                axs[1].plot(time, value['separation_speed'], linewidth=1.5, label=key)
                axs[2].plot(time, value['slip_speed'], linewidth=1.5, label=key)
        axs[0].set_ylabel('Penetration \nDepth (m)')
        axs[1].set_ylabel('Separation \nVelocity (m/s)')
        axs[2].set_ylabel('Slip \nVelocity (m/s)')
        axs[0].legend(frameon=False)
        return fig, axs

    def save_data(self, logs, savename):
        """Save the logging data to a file"""
        logs, contact = logs[0], logs[1]
        # Grab the data
        t = logs.sample_times()
        nX = self.a1.multibody.num_positions() + self.a1.multibody.num_velocities()
        x, u = logs.data()[:nX, :], logs.data()[nX:, :]
        data = {'time': t,
                'state': x,
                'control': u,
                'contact': contact}
        utils.save(savename, data)

if __name__ == '__main__':
    print("Hello from a1_simulator")