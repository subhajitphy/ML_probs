import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters for data generation
sh = int(1e6)  # Number of samples
np.random.seed(42)  # You can choose any integer
# Define low and high bounds for each dimension
lows = [0, 0, 1.0, 10, 0.1]  # [t, phi0, omega0, Mc, eta]
highs = [10, 2 * np.pi, 10.0, 80, 0.25]

# Sample uniformly from the 5D space
samples = np.random.uniform(low=lows, high=highs, size=(sh, 5))
results = np.zeros((sh, 2))  # To store [omega(t), phi(t)]

# Define the true system
def true_system(t, y, Mc, eta):
    omega, phi = y
    d_omega = -eta * omega + (1 / Mc) * np.sin(phi)
    d_phi = omega
    return [d_omega, d_phi]
def true_system_torch(omega, phi, Mc, eta):
    d_omega = -eta * omega + (1 / Mc) * torch.sin(phi)
    d_phi = omega
    return d_omega, d_phi
# Solve the system up to t
def get_PHI_omg(t0, phi0, omg0, Mc, eta, t_epoch):
    sol = solve_ivp(
        true_system,
        [t0, t_epoch],
        y0=[omg0, phi0],
        args=(Mc, eta),
        t_eval=[t_epoch],
        rtol=1e-9,
        atol=1e-9
    )
    return sol.y[1, 0], sol.y[0, 0]  # phi(t), omega(t)


# # Loop over all samples and compute the results
# for i in tqdm(range(sh), desc="Generating data"):
#     t, phi0_, omg0_, Mc_, eta_ = samples[i]
#     phi_val, omg_val = get_PHI_omg(t0=0, phi0=phi0_, omg0=omg0_, Mc=Mc_, eta=eta_, t_epoch=t)
#     results[i, 0] = omg_val
#     results[i, 1] = phi_val
import multiprocessing as mp

def solve_single(sample):
    t, phi0, omg0, Mc, eta = sample
    phi, omg = get_PHI_omg(t0=0, phi0=phi0, omg0=omg0, Mc=Mc, eta=eta, t_epoch=t)
    return [omg, phi]

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_list = list(tqdm(pool.imap(solve_single, samples), total=sh))

    results = np.array(results_list)
# Reorder input: [t, Mc, eta, omega0, phi0]
X_data = samples[:, [0, 3, 4, 2, 1]]  # [t, Mc, eta, omega0, phi0]
Y_data = results  # [omega(t), phi(t)]

# Convert to torch tensors (float32)
X_data = torch.tensor(X_data, dtype=torch.float32)
Y_data = torch.tensor(Y_data, dtype=torch.float32)
data_tensor = torch.cat([X_data, Y_data], dim=1)
torch.save(data_tensor, "./data_test_1e6.pt")
#from google.colab import files

# Download the file
#files.download("data_test_1e6.pt")
