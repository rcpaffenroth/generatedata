"""
Data generation functions for synthetic and real datasets.
"""
import json
import numpy as np
import pandas as pd
import torch
import requests
import pickle
from pathlib import Path
from torchvision import datasets, transforms
from generatedata.save_data import save_data
import mnist1d


def create_info_json(data_dir: Path) -> None:
    """Create an empty info.json file in the processed data directory."""
    info = {}
    with open(data_dir / 'info.json', 'w+') as f:
        json.dump(info, f)

def generate_regression_line(data_dir: Path, num_points: int = 1000) -> None:
    x_on = np.random.uniform(0, 1, num_points)
    y_on = 0.73 * x_on
    x_off = x_on
    y_off = y_on + np.random.normal(0, 0.1, num_points)
    start_data = {f'x{i}': arr for i, arr in enumerate([x_off, y_off])}
    target_data = {f'x{i}': arr for i, arr in enumerate([x_on, y_on])}
    save_data(data_dir, 'regression_line', start_data, target_data, x_y_index=1)

def generate_pca_line(data_dir: Path, num_points: int = 1000) -> None:
    rho = np.random.uniform(0, 1, num_points)
    theta = 0.54
    x_on = rho * np.cos(theta)
    y_on = rho * np.sin(theta)
    gamma = np.random.normal(0, 0.1, num_points)
    x_off = x_on + gamma * np.cos(theta + np.pi / 2.0)
    y_off = y_on + gamma * np.sin(theta + np.pi / 2.0)
    start_data = {f'x{i}': arr for i, arr in enumerate([x_off, y_off])}
    target_data = {f'x{i}': arr for i, arr in enumerate([x_on, y_on])}
    save_data(data_dir, 'pca_line', start_data, target_data)

def generate_circle(data_dir: Path, num_points: int = 1000) -> None:
    thetas = np.random.uniform(0, 2 * np.pi, num_points)
    x_on = np.cos(thetas)
    y_on = np.sin(thetas)
    r_off = np.random.uniform(0.8, 1.2, num_points)
    x_off = r_off * np.cos(thetas)
    y_off = r_off * np.sin(thetas)
    start_data = {f'x{i}': arr for i, arr in enumerate([x_off, y_off])}
    target_data = {f'x{i}': arr for i, arr in enumerate([x_on, y_on])}
    save_data(data_dir, 'circle', start_data, target_data)

def generate_regression_circle(data_dir: Path, num_points: int = 1000) -> None:
    thetas = np.random.uniform(0, 2 * np.pi, num_points)
    x_on = np.cos(thetas)
    y_on = np.sin(thetas)
    y_noise = np.random.normal(0, 0.2, num_points)
    x_off = np.cos(thetas)
    y_off = np.sin(thetas) + y_noise
    start_data = {f'x{i}': arr for i, arr in enumerate([x_off, y_off])}
    target_data = {f'x{i}': arr for i, arr in enumerate([x_on, y_on])}
    save_data(data_dir, 'regression_circle', start_data, target_data, x_y_index=1)

def swiss_roll(n_samples: int = 1000) -> np.ndarray:
    u = np.random.rand(n_samples)
    v = np.random.rand(n_samples)
    x = np.cos(u * 2 * np.pi) * (1 - 0.5 * u)
    y = np.sin(u * 2 * np.pi) * (1 - 0.5 * u)
    z = v
    return np.column_stack((x, y, z))

def generate_manifold(data_dir: Path, num_points: int = 1000) -> None:
    x_on = swiss_roll(num_points)
    x_off = x_on + np.random.normal(0, 0.1, (num_points, 3))
    start_data = {f'x{i}': x_off[:, i] for i in range(x_off.shape[1])}
    target_data = {f'x{i}': x_on[:, i] for i in range(x_on.shape[1])}
    save_data(data_dir, 'manifold', start_data, target_data)

def mnist1d_save_data(data_dir, name, num_points, mnist1d_dataset,
                                vector_dim=40,
                                additional_info=None):
    # Create a list of PyTorch tensors containing the MNIST digits
    digit_tensors = []
    for i in range(num_points):
        digit_tensor, label = torch.tensor(mnist1d_dataset['x'][i]), mnist1d_dataset['y'][i]

        # Append the classification label as a one-hot encoding to the end of the tensor
        one_hot = torch.zeros(10)
        one_hot[label] = 1
        # template = torch.tensor(mnist1d_dataset['templates']['x'][label,:])
        # digit_tensor = torch.cat((digit_tensor, template, one_hot))
        digit_tensor = torch.cat((digit_tensor, one_hot))

        # Append the tensor to the list of digit tensors
        digit_tensors.append(digit_tensor)

    # x_on contains the MNIST digits with the correct classification
    x_on = torch.stack(digit_tensors)
    # x_ff contains the MNIST digits with a random classification
    x_off = torch.stack(digit_tensors)
    # x_off[:, -10:] = torch.rand(size=(num_digits,10))
    x_off[:, -10:] = 0.1

    start_data = {f'x{i}': x_off[:,i] for i in range(x_off.shape[1])}
    target_data = {f'x{i}': x_on[:,i] for i in range(x_on.shape[1])}
    save_data(data_dir, name, start_data, target_data, 
                x_y_index=vector_dim, additional_info=additional_info)

def generate_mnist1d(data_dir: Path, num_points: int = 1000) -> None:
    # A simpler version of the MNIST dataset based on the work of Greydanus et al. (2018)
    # https://arxiv.org/abs/2011.14439

    url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
    r = requests.get(url, allow_redirects=True)

    mnist1d_dataset = pickle.loads(r.content)
    mnist1d_save_data(data_dir, 'MNIST1D', num_points, mnist1d_dataset) 

def generate_mnist1d_custom(data_dir: Path, num_points: int = 1000,
                            scale_coeff=0.4, max_translation=48,
                            corr_noise_scale=0.25, iid_noise_scale=2e-2,
                            shear_scale=0.75) -> None:
    # These are the default parameters used in the original MNIST1D dataset
    # arg_dict = {'num_samples': 5000,
    #             'train_split': 0.8,
    #             'template_len': 12,
    #             'padding': [36,60],
    #             # Ones that are interesting to change
    #             'scale_coeff': .4, 
    #             'max_translation': 48,
    #             'corr_noise_scale': 0.25,
    #             'iid_noise_scale': 2e-2,
    #             'shear_scale': 0.75,
    #             #  
    #             'shuffle_seq': False,
    #             'final_seq_length': 40,
    #             'seed': 42}

    arg_dict = {'num_samples': 5000,
                'train_split': 0.8,
                'template_len': 12,
                'padding': [36,60],
                # Ones that are interesting to change
                'scale_coeff': scale_coeff, 
                'max_translation': max_translation,
                'corr_noise_scale': corr_noise_scale,
                'iid_noise_scale': iid_noise_scale,
                'shear_scale': shear_scale,
                #  
                'shuffle_seq': False,
                'final_seq_length': 40,
                'seed': 42}

    mnist1d_dataset = mnist1d.data.make_dataset(mnist1d.utils.ObjectView(arg_dict))

    name = f'MNIST1Dcustom_scale{scale_coeff}_maxtrans{max_translation}_corrnoise{corr_noise_scale}_iidnoise{iid_noise_scale}_shear{shear_scale}'
    mnist1d_save_data(data_dir, name, num_points, mnist1d_dataset, additional_info=arg_dict) 

def mnist_save_data(data_dir, name, num_points, mnist_dataset,
                    vector_dim=28 * 28,
                    additional_info=None):
    digit_tensors = []
    # Get the dimension of the label
    label_dim = mnist_dataset.targets.max()+1
    for _ in range(num_points):
        random_image, label = mnist_dataset[np.random.randint(len(mnist_dataset))]
        digit_tensor = random_image.view(-1)
        one_hot = torch.zeros(label_dim)
        one_hot[label] = 1
        digit_tensor = torch.cat((digit_tensor, one_hot))
        digit_tensors.append(digit_tensor)
    x_on = torch.stack(digit_tensors)
    x_off = torch.stack(digit_tensors)
    x_off[:, -label_dim:] = 0.1
    start_data = {f'x{i}': x_off[:, i] for i in range(x_off.shape[1])}
    target_data = {f'x{i}': x_on[:, i] for i in range(x_on.shape[1])}
    save_data(data_dir, name, start_data, target_data, x_y_index=vector_dim, additional_info=additional_info)
    
def generate_mnist(data_dir: Path, num_points: int = 1000) -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_dataset = datasets.MNIST(root=str(data_dir.parent.parent / 'external'), train=True, download=True, transform=transform)
    mnist_save_data(data_dir, 'MNIST', num_points, mnist_dataset, vector_dim=28*28)

def generate_mnist_custom(data_dir: Path, num_points: int = 1000,
                          dataset_name = 'MNIST',
                          degrees=(0,20), translate=(0.0, 0.2), scale=(0.7, 0.9)) -> None:
    """dataset_name can be 'MNIST', 'EMNIST', 'KMNIST', 'FashionMNIST'"""
    dataset_cls = getattr(datasets, dataset_name)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((50, 50)),  # Resize to 50x50 for consistency
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale) # Make the problem harder
    ])
    additional_info = {
        'dataset_name': dataset_name,
        'degrees': degrees,
        'translate': translate,
        'scale': scale
    }
    if dataset_name == "EMNIST":
        mnist_dataset = dataset_cls(root="data/external", train=True, download=True, transform=transform, split="letters")
    else:
        mnist_dataset = dataset_cls(root="data/external", train=True, download=True, transform=transform)
    name = f"{dataset_name}_custom_degrees{degrees[0]}_{degrees[1]}_translate{translate[0]}_{translate[1]}_scale{scale[0]}_{scale[1]}"
    mnist_save_data(data_dir, name, num_points, mnist_dataset, vector_dim=50*50, additional_info=additional_info)
               
def generate_emlocalization(data_dir: Path) -> None:
    X = torch.load(data_dir.parent / 'raw' / 'EM_X_train.pt')
    Y = torch.load(data_dir.parent / 'raw' / 'EM_Y_train.pt')
    x_on = torch.cat((X, Y), dim=1)
    y_max = x_on[:, -1].max()
    y_min = x_on[:, -1].min()
    x_on[:, -1] = (x_on[:, -1] - y_min) / (y_max - y_min)
    x_off = x_on.clone()
    x_off[:, -1] = 0
    start_data = {f'x{i}': x_off[:, i] for i in range(x_off.shape[1])}
    target_data = {f'x{i}': x_on[:, i] for i in range(x_on.shape[1])}
    save_data(data_dir, 'EMlocalization', start_data, target_data, x_y_index=160)

def generate_lunarlander(data_dir: Path) -> None:
    lunarlander_df = pd.read_parquet(data_dir.parent / 'raw' / 'lander_all_data.parquet')
    x_off = []
    x_on = []
    for i, state in enumerate(['random', 'trained', 'good', 'better']):
        X = lunarlander_df.loc[(state, slice(None)), (slice(None), ('x', 'y', 'vx', 'vy'))]
        Y_off = torch.zeros((X.shape[0], 4))
        Y_on = torch.zeros((X.shape[0], 4))
        Y_on[:, i] = 1
        x_off.append(torch.cat((torch.tensor(X.values), Y_off), dim=1))
        x_on.append(torch.cat((torch.tensor(X.values), Y_on), dim=1))
    x_off = torch.cat(x_off, dim=0)
    x_on = torch.cat(x_on, dim=0)
    start_data = {f'x{i}': x_off[:, i] for i in range(x_off.shape[1])}
    target_data = {f'x{i}': x_on[:, i] for i in range(x_on.shape[1])}
    save_data(data_dir, 'LunarLander', start_data, target_data, x_y_index=404)

def generate_massspec(data_dir: Path) -> None:
    mass_spec_df = pd.read_parquet(data_dir.parent / 'raw' / 'mass_spec.parquet')
    new_order = list(mass_spec_df.columns[:915]) + list(mass_spec_df.columns[1427:-1]) + list(mass_spec_df.columns[915:1427])
    start_data = mass_spec_df[new_order]
    target_data = mass_spec_df[new_order]
    pd.options.mode.copy_on_write = True
    start_data.iloc[:, -512:] = 0.0
    save_data(data_dir, 'MassSpec', start_data, target_data, x_y_index=1433-512)

def generate_all(data_dir: Path) -> None:
    create_info_json(data_dir)
    generate_regression_line(data_dir)
    generate_pca_line(data_dir)
    generate_circle(data_dir)
    generate_regression_circle(data_dir)
    generate_manifold(data_dir)
    generate_mnist1d(data_dir)
    generate_mnist1d_custom(data_dir, iid_noise_scale=4e-2)
    generate_mnist_custom(data_dir, dataset_name='EMNIST', degrees=(0, 10))
    generate_mnist(data_dir)
    generate_emlocalization(data_dir)
    generate_lunarlander(data_dir)
    generate_massspec(data_dir)
