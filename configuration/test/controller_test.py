import os

from pydrake.all import ContactModel, DiagramBuilder, MultibodyPlant, Parser, SceneGraph

from configuration.build_from_config import build_from_config
from configuration.controller import ContactEILControllerConfig, MPCControllerConfig
from drake_simulation import controllers

URDF = os.path.join(
    "pycito", "systems", "A1", "A1_description", "urdf", "a1_foot_collision.urdf"
)
REFERENCE = os.path.join("data", "a1", "reference", "symmetric", "3m", "reftraj.pkl")


def sample_plant():
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = builder.AddSystem(MultibodyPlant(time_step=0.01))
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    model = Parser(plant=plant).AddModelFromFile(URDF, "quad")
    plant.set_contact_model(ContactModel.kPoint)
    plant.Finalize()
    return plant


def test_build_mpc_controller() -> None:
    config = MPCControllerConfig(timestep=0.01, reference_path=REFERENCE, horizon=5)
    controller = build_from_config(controllers, config, plant=sample_plant())
    assert isinstance(controller, controllers.A1ContactMPCController)


def test_build_contact_eil_controller() -> None:
    config = ContactEILControllerConfig(
        timestep=0.01, reference_path=REFERENCE, horizon=5
    )
    controller = build_from_config(controllers, config, plant=sample_plant())
    assert isinstance(controller, controllers.A1ContactEILController)
