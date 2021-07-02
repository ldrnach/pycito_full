"""
Run all A1 optimization configurations saved in "examples/a1/runs"

"""

import os
from datetime import date
from trajopt.optimizer import A1VirtualBaseOptimizer

def make_output_directory(configfile):
    """Create the output directory for the current run"""
    # Split the directory from the filename
    directory = os.path.dirname(configfile)
    filename = os.path.basename(configfile)
    # Use the current date as part of the filename
    datestr = date.today().strftime("%b-%d-%Y")
    targetdir = os.path.join(directory, datestr)
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)
    # Make the subdirectory for this run - increment the number until a new directory needs to be created
    subdir = "run"
    n = 1
    while os.path.isdir(os.path.join(targetdir, f"{subdir}_{n}")):
        n+=1
    # Make the new subdirectory
    target = os.path.join(targetdir, f"{subdir}_{n}")
    os.mkdir(target)
    # Return the target directory and the filename
    return target, filename

def run_optimization(configfile):
    """Run the optimization routine"""
    # Load the configuration and create the optimizer
    optimizer = A1VirtualBaseOptimizer.buildFromConfig(configfile)
    # Finalize & run the program
    optimizer.finalizeProgram()
    result = optimizer.solve()
    # Create a directory for saving the outputs
    directory, filename = make_output_directory(configfile)    
    # Save the results as an array
    optimizer.saveResults(result, name=os.path.join(directory,  "trajoptresults.pkl"))
    optimizer.saveReport(result, savename=os.path.join(directory, "report.txt"))
    # Save the figures
    optimizer.saveDebugFigure(savename = os.path.join(directory, "CostsAndConstraints.png"))
    optimizer.plot(result, show=False, savename=os.path.join(directory, "trajopt.png"))
    optimizer.plotConstraints(result, show=False, savename=os.path.join(directory, "debug.png"))

    # Move the configuration file to the new directory
    os.rename(configfile, os.path.join(directory, filename))
    # Return the status of the optimization
    return result.is_success()

if __name__ == "__main__":
    dirname = os.path.join("examples", "a1", "runs")
    files = [file for file in os.listdir(dirname) if file.endswith(".pkl")]
    num_files = len(files)
    n = 1
    successes = 0
    for file in files:
        print(f"Running file {n} of {num_files}")
        status = run_optimization(os.path.join(dirname, file))
        print(f"Completed successfully? {status}")
        if status:
            successes += 1
    print(f"{successes} of {num_files} solved successfully")