import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nas_architecture import DARTSModel, DARTSModelLayers
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import pandas as pd
import os
import json
import pickle
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

gpu_name = torch.cuda.get_device_name(device=None)

def plot_losses(train_loss, valid_loss, epochs):
    plt.figure()
    plt.plot(epochs, train_loss, '-b')  # Solid blue line for MSE loss
    plt.plot(epochs, valid_loss, '-r')  # Solid red line for sparsity loss
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train loss', 'valid loss'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(figure_location+'loss_vs_epochs.png')
    plt.close()

def create_dataset(num_samples, n, m, s):
    rng_seed = 42
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    W = np.random.random([n, m])
    Wd = 10 * W / np.linalg.norm(W)
    full_Wd = np.tile(Wd, (num_samples, 1, 1))
    z = np.zeros([num_samples, m])
    for z_sample in z:
        z_sample[np.random.randint(m, size=s)] = (1 - (-1)) * np.random.random_sample([s]) + (-1)
    full_z = np.expand_dims(z, 2)
    full_x = np.matmul(full_Wd, full_z)
    return torch.FloatTensor(full_x).to(device), torch.FloatTensor(Wd).to(device), torch.FloatTensor(full_z).to(device)

def train_darts(model, train_data, valid_data, epochs, lr_w, lr_a):
    criterion = nn.MSELoss()
    arch_params = [p for name, p in model.named_parameters() if 'alpha' in name]
    optimizer_a = optim.Adam(arch_params, lr=lr_a)

    train_loader = DataLoader(train_data, batch_size=600, shuffle=True)

    for epoch in range(epochs):

        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer_a.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer_a.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            test_x = valid_data[0][0].expand(1, 50, 1).to(device)
            model.eval()
            test_z = model(test_x)
            real_z = valid_data[0][1].to(device)
            plt.figure()
            plt.plot(test_z.detach().cpu().numpy().squeeze(), label='nas')  
            plt.title('Sparse Signal NAS @ Epoch ' + str(epoch))
            plt.legend()
            plt.savefig(figure_location + "epoch_figures/ground_truth_nas" + str(epoch) + ".png")
            plt.close()

            plt.figure()
            plt.plot(real_z.cpu(), label='original')
            plt.title('Sparse Signal Original')
            plt.legend()
            plt.savefig(figure_location + "epoch_figures/ground_truth_nas" + str(epoch) + "og.png")
            plt.close()

            plt.figure()
            plt.plot(test_z.detach().cpu().numpy().squeeze() - real_z.cpu().numpy().squeeze())
            plt.title('Error in Sparse Signal Reconstruction @ Epoch ' + str(epoch))
            plt.ylim(-1, 1)
            plt.savefig(figure_location + "epoch_figures/error" + str(epoch) + ".png")
            plt.close()

            plt.figure()
            labels = ["shrinkage", "relu", "identity", "gelu"]
            df = pd.DataFrame(model.alpha.detach().cpu().numpy().transpose(), index=labels)
            sns.heatmap(df, cmap='coolwarm')
            plt.title("NAS Alpha Values @ Epoch " + str(epoch))
            plt.ylabel("Operation")
            plt.xlabel("Layer Number")
            plt.savefig(figure_location + "epoch_figures/heatmap" + str(epoch) + ".png")
            plt.close()

            plt.figure()
            labels = ["shrinkage", "relu", "identity", "gelu"]
            df = pd.DataFrame(model.alpha.grad.detach().cpu().numpy().transpose(), index=labels)
            sns.heatmap(df, cmap='coolwarm')
            plt.title("Alpha Gradient Values @ Epoch " + str(epoch))
            plt.ylabel("Operation")
            plt.xlabel("Layer Number")
            plt.savefig(figure_location + "epoch_figures/grad_heatmap" + str(epoch) + ".png")
            plt.close()
    return model

def train_darts_layer_count(model, train_data, valid_data, epochs, lr_w, lr_a, history):
    criterion = nn.MSELoss()
    alpha_intermediate = []
    train_loss = []
    valid_loss = []
    arch_params = [p for name, p in model.named_parameters() if 'alpha' in name]
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer_a = optim.Adam(arch_params, lr=lr_a)

    train_loader = DataLoader(train_data, batch_size=20000, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=20000, shuffle=False)

    overall_start_time = time.perf_counter()

    for epoch in tqdm(range(epochs)):
        alpha_intermediate.append(model.alpha.data.detach().clone())
        train_epoch_loss = 0
        valid_epoch_loss = 0
        start_time = time.time()
        model.train()
        model.eval_flag = False
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer_a.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer_a.step()
                train_epoch_loss += loss.item()
        
        model.eval()
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            valid_epoch_loss += loss.item()
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        
        if epoch % 50 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer_a.state_dict(),
                'epoch': epoch,
            }
            with open(figure_location + 'saved_pickle/train_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)
            with open(figure_location + 'saved_pickle/valid_loss.pkl', 'wb') as file:
                pickle.dump(valid_loss, file)
            with open(figure_location + 'saved_pickle/alpha_intermediate.pkl', 'wb') as file:
                pickle.dump(alpha_intermediate, file)
            
            with open('loss_history.json', 'w') as json_file:
                json.dump(history, json_file, indent=4)
            
            plot_losses(train_loss, valid_loss, history['epochs'])

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            test_x = valid_data[0][0].expand(1, model.input_size, 1).to(device)
            model.eval()
            model.eval_flag = False
            test_z = model(test_x)
            real_z = valid_data[0][1].to(device)
            plt.figure()
            plt.plot(test_z.detach().cpu().numpy().squeeze(), label='nas')  
            plt.title('Sparse Signal NAS @ Epoch ' + str(epoch))
            plt.legend()
            plt.savefig(figure_location + "signal_figures/ground_truth_nas" + str(epoch) + ".png")
            plt.close()

            plt.figure()
            plt.plot(real_z.cpu(), label='original')
            plt.title('Sparse Signal Original')
            plt.legend()
            plt.savefig(figure_location + "signal_figures/ground_truth_nas" + str(epoch) + "og.png")
            plt.close()

            plt.figure()
            plt.plot(test_z.detach().cpu().numpy().squeeze() - real_z.cpu().numpy().squeeze())
            plt.title('Error in Sparse Signal Reconstruction @ Epoch ' + str(epoch))
            plt.ylim(-1, 1)
            plt.savefig(figure_location + "signal_figures/error" + str(epoch) + ".png")
            plt.close()

            plt.figure()
            labels = ["Shrinkage", "relu", "Identity", "gelu", "elu", "Hardshrink", "Hardtanh", "Hardswish", "selu", "celu", "LeakyRelu", "LogSigmoid", "Tanhshrink", "Tanh", "sigmoid", "Hardsigmoid", "silu", "Mish"]
            df = pd.DataFrame(model.alpha.detach().cpu().numpy().transpose(), index=labels)
            sns.heatmap(df, cmap='coolwarm')
            plt.title("NAS Alpha Values @ Epoch " + str(epoch))
            plt.ylabel("Operation")
            plt.xlabel("Layer Number")
            plt.savefig(figure_location + "alpha_figures/alpha_heatmap" + str(epoch) + ".png")
            plt.close()

            plt.figure()
            labels = ["Shrinkage", "relu", "Identity", "gelu", "elu", "Hardshrink", "Hardtanh", "Hardswish", "selu", "celu", "LeakyRelu", "LogSigmoid", "Tanhshrink", "Tanh", "sigmoid", "Hardsigmoid", "silu", "Mish"]
            df = pd.DataFrame(model.alpha.grad.detach().cpu().numpy().transpose(), index=labels)
            sns.heatmap(df, cmap='coolwarm')
            plt.title("Alpha Gradient Values @ Epoch " + str(epoch))
            plt.ylabel("Operation")
            plt.xlabel("Layer Number")
            plt.savefig(figure_location + "alpha_figures/alpha_grad_heatmap" + str(epoch) + ".png")
            plt.close()

        end_time = time.time()
        print("Epoch Time:" + str(end_time - start_time))
        history['train_losses'].append(train_epoch_loss)
        history['epochs'].append(epoch+1)
        print(f"epoch: {epoch}, train_loss: {train_epoch_loss:.6f}")
        if epoch%50 == 0:
            plot_losses(train_loss, valid_loss, history['epochs'])
    total_time_taken = time.perf_counter() - overall_start_time
    print(f"total epochs: {epochs}, total time taken: {total_time_taken:.4f}")
    return model, history, train_loss, valid_loss, alpha_intermediate


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process a job ID as a string from the command line.")

    # Add an argument for the job_id input, setting the type to string (str)
    parser.add_argument("--job_id", type=int, help="a job ID as a str input")

    # Parse the command-line arguments
    args = parser.parse_args()

    job_id = args.job_id

    figure_location = f"/scratch/wpy2004/lista/report_jan_data/{job_id}/"
    print(figure_location)
    os.makedirs(os.path.dirname(figure_location), exist_ok=True)
    os.makedirs(os.path.dirname(figure_location + "alpha_figures/"), exist_ok=True)
    os.makedirs(os.path.dirname(figure_location + "signal_figures/"), exist_ok=True)
    os.makedirs(os.path.dirname(figure_location + "saved_pickle/"), exist_ok=True)
        
    num_samples = 5000
    
    if job_id == 0:
        n, m = 25, 50
        epochs = 1001
        lr_w, lr_a = 0.05, 0.05
        num_iterations = 600
        s = 5
    elif job_id == 1:
        n, m = 25, 50
        epochs = 1001
        lr_w, lr_a = 0.1, 0.1
        num_iterations = 600
        s= 5

    input_size, hidden_size = n, m
    history = {
        'train_losses':[],
        'epochs':[],
    }
        
    X, Wd, Z = create_dataset(num_samples, n, m, s)
    Wd = Wd.reshape(1, Wd.shape[0], Wd.shape[1])
    train_size = int(0.8 * num_samples)
    train_data = TensorDataset(X[:train_size], Z[:train_size])
    valid_data = TensorDataset(X[train_size:], Z[train_size:])
    
    binary_model = DARTSModel(input_size, hidden_size, Wd, num_iterations).to(device)
    
    binary_model, history, train_loss, valid_loss, alpha_intermediate = train_darts_layer_count(binary_model, train_data, valid_data, epochs, lr_w, lr_a, history)
    
    torch.save(binary_model.state_dict(), figure_location + "saved_pickle/fista_nas.pth")
    with open(figure_location + 'saved_pickle/train_loss.pkl', 'wb') as file:
        pickle.dump(train_loss, file)
    with open(figure_location + 'saved_pickle/valid_loss.pkl', 'wb') as file:
        pickle.dump(valid_loss, file)
    with open(figure_location + 'saved_pickle/alpha_intermediate.pkl', 'wb') as file:
        pickle.dump(alpha_intermediate, file)
    
    with open('loss_history.json', 'w') as json_file:
        json.dump(history, json_file, indent=4)
    
    plot_losses(train_loss, valid_loss, history['epochs'])
    
