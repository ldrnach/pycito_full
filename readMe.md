## PYCITO: Contact Implicit Trajectory Optimization in Python, using pyDrake

This project is currently in development. 

### Overview
This project contains utilities for robot motion planning with intermittent frictional contact. We use pyDrake as a backend for calculating rigid body dynamics and for solving general nonlinear programming problems. We also use and extend the trajectory optimization tools and implement contact-implicit trajectory optimization in Python.

+ Author: Luke Drnach
+ Affiliation: [The LIDAR Lab](http://lab-idar.gatech.edu/), Georgia Institute of Technology

### Docker Container Notes
This project is being developed inside a [docker container with Drake installed](https://hub.docker.com/r/robotlocomotion/drake). The setup for the container is contained with the directory .devcontainer, which also sets up X-11 forwarding on a Windows host. I have been developing within the container in VSCode running on a Windows 10 host machine. With some modification, the setup also works on Ubuntu-18.04 hosts.

### Project Directories
The implementation of contact implicit trajectory optimization and the examples are separated across multiple directories. The primary directories are *trajopt*, *systems*, and *examples*. All three directories are still in development

In *trajopt*:
+ *contactimplicit.py* contains a generic implementation of contact-implicit trajectory optimziation. More advanced implementations are in development on other branches.

In *systems*:
+ *timestepping.py* extends rigid multibody dynamics to systems involving intermittent contact using a time-stepping approach. Specific examples of timestepping systems include a sliding block and the A1 quadruped, available in the *block* and *A1* subdirectories. 

+ *terrain.py* contains a generic implementation for representing terrains using surface models. This is an alternative to using URDF models of terrains, and allows us to build up towards implicit representations of terrain models, for example, using Gaussian Process Regression.

In *examples*:
+ *examples/sliding_block* contains example scripts for running trajectory optimization on a planar sliding block
+ *examples/a1* contains example scripts for running trajectory optimization on the A1 quadruped. This example is still in development
+ *helloDrake.py*: a simple example to check that Drake is properly installed
+ *fallingRod.py*: a time-stepping simulation example based on an old implementation. 

The directory *unittests* contains a handful of unittests checking the implemenation of several parts of the pyCITO project; however, at this time, the unit tests are not exhaustive. 

### pyDrake tutorials
To better understand the interface with pyDrake, pyCITO contains several tutorials on Drake's MultibodyPlant and MathematicalProgram classes. The *tutorials* directory contains several tutorials on the Drake methods available in pyDrake and how to use them. 

In *tutorials/multibody*:
+ **kinematics.py** details accessing rigid body frames and performing forward kinematics using MultibodyPlant
+ **dynamics.py** details accessing the dynamic properties - such as the generalized mass and coriolis matrices - from MultibodyPlant, setting external forces, and performing inverse dynamics
+ **collision.py** details extracting collision geometries from MultibodyPlant

In *tutorials/optimization*:
+ **mathprog.py** details setting up a simple optimization problem with Drake's MathematicalProgram
+ **doubleIntegrator.py** details setting up a trajectory optimization problem on a linear system with custom constraints, using MathematicalProgram
+ **acrobotSwingUp.py** details setting up a trajectory optimization problem using MultibodyPlant and DirectCollocation