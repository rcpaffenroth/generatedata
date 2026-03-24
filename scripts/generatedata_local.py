import argparse
import pathlib
import generatedata
from generatedata.data_generators import generate_all

# This script is not just for testing!  This script is what is used to generate the data for the project.
# It is not meant to be run on a regular basis, but it is run when a new data set is needed.

def main(all=False, lra=False):
    base_dir = pathlib.Path(generatedata.__path__[0])
    data_dir = base_dir / '../data/processed'
    generate_all(data_dir, all=all, lra=lra)

def test_main(all=False):
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets for the generatedata library.")
    parser.add_argument("--all", action="store_true", help="Generate full parameter sweeps.")
    parser.add_argument("--lra", action="store_true", help="Generate Long Range Arena (LRA) benchmark datasets.")
    args = parser.parse_args()
    main(all=args.all, lra=args.lra)
