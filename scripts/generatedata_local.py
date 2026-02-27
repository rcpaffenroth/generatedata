import pathlib
import generatedata
from generatedata.data_generators import generate_all


def main(all=False):
    base_dir = pathlib.Path(generatedata.__path__[0])
    data_dir = base_dir / '../data/processed'
    generate_all(data_dir, all=all)

def test_main(all=False):
    main()

if __name__ == "__main__":
    main(all=True)
