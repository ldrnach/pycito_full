import json, os

filename = os.path.join('hardwareinterface','estimatorconfig.json')

configuration = {
    'Estimator': {
        'Horizon': 5,
        'ForceCost': 1e2,
        'RelaxationCost': 1e3,
        'DistanceCost': 1,
        'FrictionCost': 1,
        'VelocityScaling': 1e-3,
        'ForceScaling': 1e2,
        'EnableLogging': False,
        'InitialLegPose': [0.0, 0.8, -1.6]
    },
    'Solver':{
        'Major feasibility tolerance': 1e-6,
        'Major optimality tolerance': 1e-6
    },
    'FrictionModel':{
        'Prior': 1.0,
        'KernelRegularization': 1.0
    },
    'SurfaceModel':{
        'PriorLocation': 0,
        'PriorDirection': [0., 0., 1.],
        'KernelWeights': [0.01, 0.01, 0.01],
        'KernelRegularization': 0.001
    }
}

if __name__ == '__main__':
    with open(filename, 'w') as file:
        json.dump(configuration, file, indent=4, sort_keys=True)