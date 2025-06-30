from generatedata import load_data

def test_data_names_local():
    print(load_data.data_names(local=True))

def test_load_data_local():
    print(load_data.load_data('MNIST', local=True))

def test_data_names_remote():
    print(load_data.data_names(local=False))

def test_load_data_remote():
    print(load_data.load_data('MNIST', local=False))


