"""
Collect block results for Zhigen

"""
import os
import pycito.utilities as utils

def find_pickle_recursive(source):
    for path, dir, files in os.walk(source):
        for file in files:
            if file == "trajoptresults.pkl":
                yield os.path.join(path, file)

def main():
    directory = os.path.join("examples", "sliding_block", "runs","Aug-23-2021")
    files = [file for file in find_pickle_recursive(directory)]
    files = utils.alphanumeric_sort(files)
    dataset = []
    data = utils.load(files[0])
    print(data.keys())
    for file in files:
        data = utils.load(file)
        if data['success']:
            data['filename'] = file
            dataset.append(data)
    filename = os.path.join('examples','sliding_block','runs','blockdata.pkl')
    print(f"Saved {len(dataset)} files at {filename}")
    utils.save(filename, dataset)



if __name__ == "__main__":
    main()