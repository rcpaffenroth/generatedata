import argparse
import pathlib
import generatedata
from generatedata.data_generators import generate_all
from generatedata.whest_generators import generate_whest_all

# This script is not just for testing!  This script is what is used to generate the data for the project.
# It is not meant to be run on a regular basis, but it is run when a new data set is needed.

def main(all=False, whest=False, whest_xl=False):
    base_dir = pathlib.Path(generatedata.__path__[0])
    data_dir = base_dir / '../data/processed'
    generate_all(data_dir, all=all)
    # The whest ladder is opt-in: unlike everything above it wants a GPU, downloads
    # about a gigabyte of officially published competition data, and writes hundreds
    # of megabytes.  --whest-xl adds two datasets at five times the size budget.
    if whest or whest_xl:
        generate_whest_all(data_dir, include_xl=whest_xl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets for the generatedata library.")
    parser.add_argument("--all", action="store_true", help="Generate full parameter sweeps.")
    parser.add_argument("--whest", action="store_true",
                        help="Also generate the whest ladder (~680 MB; wants a GPU and "
                             "downloads ~1 GB of official competition data).")
    parser.add_argument("--whest-xl", action="store_true",
                        help="Also generate the two 500 MB whest datasets. Implies --whest "
                             "and takes tens of minutes.")
    args = parser.parse_args()
    main(all=args.all, whest=args.whest, whest_xl=args.whest_xl)
