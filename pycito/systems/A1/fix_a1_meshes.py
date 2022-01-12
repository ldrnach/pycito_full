import sys, os, string

def asciifilter(str):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, str))

def main():
    path = "systems/A1/A1_description/meshes/"
    files = ["calf", "hip", "thigh_mirror","thigh", "trunk"]
    ext = ".obj"
    tag = "_old"
    for file in files:
        print(f"re-encoding file {file}")
        os.rename(path + file + ext, path + file + tag + ext)
        with open(path + file + tag + ext, "r", encoding="utf-8") as f:
            lines = f.read()
        # Filter non-ascii characters
        ascii_lines = asciifilter(lines)
        with open(path + file + ext, "w") as f:
            f.write(ascii_lines)
        print('finished')

if __name__ == "__main__":
    main()