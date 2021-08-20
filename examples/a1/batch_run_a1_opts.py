"""
Run all A1 optimization configurations saved in "examples/a1/runs"

"""

#TODO: Check setting major iterations limit - NOTE, the callback is called MORE OFTEN than the major iterations. 
#TODO: Determine how to accurately count the number of major iterations and set appropriatly to prevent 30hr runs
#TODO: Serialize sequential optimizations within optimizer
#TODO: Parallelize - Technically done

import os, concurrent.futures
from matplotlib import pyplot as plt
from datetime import date
from trajopt.optimizer import A1VirtualBaseOptimizer
import utilities as utils

# Use the current date as part of the filename
datestr = date.today().strftime("%b-%d-%Y")

def make_output_directory(configfile):
    """Create the output directory for the current run"""
    # Split the directory from the filename
    directory = os.path.dirname(configfile)
    filename = os.path.basename(configfile)
    # Use the current date as part of the filename
    targetdir = os.path.join(directory, datestr)
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)
    # Make the subdirectory for this run - increment the number until a new directory needs to be created
    subdir = filename.split(".")[0]
    if os.path.isdir(os.path.join(targetdir, subdir)):
        n = 1
        while os.path.isdir(os.path.join(targetdir, f"{subdir}_({n})")):
            n += 1  
        target = os.path.join(targetdir, f"{subdir}_({n})")
    else:
        target = os.path.join(targetdir, subdir)
    os.mkdir(target)
    # Return the target directory and the filename
    return target, filename

def run_sequential_optimization(configfile):
    """ Run a sequence of optimizations"""
    print(f"Process {os.getpid()} running file {configfile}")
    configs = utils.load(configfile)
    if not isinstance(configs, list):
        return run_optimization(configfile)
    # Make the output directory
    directory, filename = make_output_directory(configfile)
    run = 1
    for config in configs:
        # Load the configuration and create the optimizer
        optimizer = A1VirtualBaseOptimizer.buildFromConfig(config)
        # Finalize the program
        optimizer.finalizeProgram()
        if run > 1:
            # Get the solution variables and re-initialize
            optimizer.useGuessFromFile(filename = os.path.join(directory, f"run_{run-1}", "trajoptresults.pkl"))
        result = optimizer.solve()
        # Save the results
        optimizer.saveResults(result, name=os.path.join(directory, f"run_{run}", "trajoptresults.pkl"))
        optimizer.saveReport(result, savename = os.path.join(directory, f"run_{run}", "report.txt"))
        # Save the figures
        optimizer.saveDebugFigure(savename = os.path.join(directory, f"run_{run}", "CostsAndConstraints.png"))
        optimizer.plot(result, show=False, savename=os.path.join(directory, f"run_{run}", "trajopt.png"))
        # Create a new guess for the next optimization
        run += 1
        plt.close('all')

    # Move the configuration file
    os.rename(configfile, os.path.join(directory, filename))
    # Close all figures and report the final results
    print(f"Process {os.getpid()} finished with file {configfile}. \nFinished successfully? {result.is_success()}")
    return result.is_success()

def run_optimization(configfile):
    """Run the optimization routine"""
    print(f"Process {os.getpid()} running file {configfile}")
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
    #optimizer.plotConstraints(result, show=False, savename=os.path.join(directory, "debug.png"))

    # Move the configuration file to the new directory
    os.rename(configfile, os.path.join(directory, filename))
    print(f"Process {os.getpid()} finished with file {configfile}. \nFinished successfully? {result.is_success()}")
    # Return the status of the optimization
    return result.is_success()

def main():
    dirname = os.path.join("examples", "a1", "runs")
    files = [os.path.join(dirname, file) for file in os.listdir(dirname) if file.endswith(".pkl")]
    files = utils.alphanumeric_sort(files)
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        successes = executor.map(run_sequential_optimization, files)
    print(f"{sum(successes)} of {len(successes)} solved successfully")

    # num_files = len(files)
    # n = 1
    # for file in files:
    #     print(f"Running file {n} of {num_files}: {file}")
    #     status = run_optimization(os.path.join(dirname, file))
    #     print(f"Completed successfully? {status}")
    #     plt.close('all')
    #     if status:
    #         successes += 1
    #     n += 1
    # print(f"{successes} of {num_files} solved successfully")

if __name__ == "__main__":
    main()