"""
Make meshcat visualizations for a sequence of a1 optimizations

Luke Drnach
July 6. 2021
"""
import os
from systems.A1.a1 import A1VirtualBase
import utilities as utils
from pydrake.all import PiecewisePolynomial

def make_a1_visualization(directory):
    """Make a visualization from a trajopt result in a directory"""
    file = os.path.join(directory, 'trajoptresults.pkl')
    if not os.path.isfile(file):
        print(f"{file} does not exist. Skipping visualization")
        return None
    # Make the visualization
    print(f"Visualizing {file}")
    data = utils.load(file)
    traj = PiecewisePolynomial.FirstOrderHold(data['time'], data['state'])
    A1VirtualBase.visualize(traj)

if __name__ == "__main__":
    dirname = os.path.join("examples","a1","runs","Jul-06-2021")
    subdirs = [f.path for f in os.scandir(dirname) if f.is_dir()]
    subdirs = utils.alphanumeric_sort(subdirs)
    for dir in subdirs:
        make_a1_visualization(dir)