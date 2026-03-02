import pathlib
import generatedata
from generatedata.data_generators import generate_all

# This script is not just for testing!  This script is what is used to generate the data for the project.  
# It is not meant to be run on a regular basis, but it is run when a new data set is needed.  

def main(all=False):
    base_dir = pathlib.Path(generatedata.__path__[0])
    data_dir = base_dir / '../data/processed'
    generate_all(data_dir, all=all)

def test_main(all=False):
    main()

if __name__ == "__main__":
    main(all=True)
