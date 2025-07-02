import pathlib
import generatedata
from generatedata.data_generators import generate_all


def main():
    base_dir = pathlib.Path(generatedata.__path__[0])
    data_dir = base_dir / '../data/processed'
    generate_all(data_dir)

def test_main():
    main()

if __name__ == "__main__":
    main()
