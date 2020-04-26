import settings
import time
import argparse
import torch
import utils
import numpy as np
from model import MDANet
import torch.optim as optim
import torch.nn.functional as F
import pickle
from load_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type = str, default="webcamT")

parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                            "not show training progress.", type=bool, default=True)
parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                    type=str, default="mdan")
# The experimental setting of using 5000 dimensions of features is according to the papers in the literature.
parser.add_argument("-d", "--dimension", help="Number of features to be used in the experiment",
                    type=int, default=5000)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1e-2)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=1)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=20)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="maxmin")
# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = utils.get_logger(args.name)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Loading the randomly partition the amazon data set.
time_start = time.time()
webcamT = load_data(10)

data_insts, data_densities, data_counts, num_insts = [], [], [], []

for id in webcamT:
    new_data_insts = []
    new_data_densities = []
    new_data_counts = []
    new_num_insts = 0
    for time_id in webcamT[id].camera_times:
        if new_num_insts > 10:
            break
        if webcamT[id].camera_times[time_id].frames[1].frame is not None:
            new_data_insts.append(webcamT[id].camera_times[time_id].frames[1].frame / 255)
            new_data_densities.append(webcamT[id].camera_times[time_id].frames[1].density)
            new_data_counts.append(len(webcamT[id].camera_times[time_id].frames[1].vehicles))
            #new_data_insts.append(webcamT[id].camera_times[time_id].frames[10].frame)
            #new_data_labels.append(len(webcamT[id].camera_times[time_id].frames[10].vehicles))
            new_num_insts += 1
    data_insts.append(new_data_insts)
    data_counts.append(new_data_counts)
    data_densities.append(new_data_densities)
    num_insts.append(new_num_insts)

##############################
################################
num_epochs = args.epoch
batch_size = args.batch_size
num_domains = settings.NUM_DATASETS - 1
lr = 0.00001
mu = args.mu
gamma = 10.0
mode = args.mode
logger.info("Training with domain adaptation using PyTorch madnNet: ")
error_dicts = {}
results = {}
for i in range(settings.NUM_DATASETS):
    # Build source instances.
    source_insts = []
    source_counts = []
    source_densities = []
    for j in range(settings.NUM_DATASETS):
        if j != i:
            array = np.array(data_insts[j], dtype=np.float)
            source_insts.append(torch.from_numpy(array).float().to(device))
            source_counts.append(torch.from_numpy(np.array(data_counts[j],  dtype=np.float)).float().to(device))
            source_densities.append(torch.from_numpy(np.array(data_densities[j],  dtype=np.float)).float().to(device))
    # Build target instances.
    target_idx = i
    target_insts = np.array(data_insts[i], dtype=np.float)
    target_densities = np.array(data_densities[i], dtype=np.float)
    target_counts = np.array(data_counts[i], dtype=np.float)
    # Train DannNet.
    mdan = MDANet(num_domains).to(device)
    optimizer = optim.Adadelta(mdan.parameters(), lr=lr)
    mdan.train()
    # Training phase.
    time_start = time.time()
    logger.info("Start training...")
    for t in range(num_epochs):
            running_loss = 0.0
            slabels = torch.ones(len(source_insts[0]), requires_grad=False).type(torch.LongTensor).to(device)
            tlabels = torch.zeros(len(source_insts[0]), requires_grad=False).type(torch.LongTensor).to(device)
            tinputs = torch.tensor(target_insts, requires_grad=False).float().to(device)
            optimizer.zero_grad()
            logger.info("Starting MDAN")
            densities, _, sdomains, tdomains = mdan(source_insts, tinputs)
            logger.info("Ending MDAN")
            # Compute prediction accuracy on multiple training sources.
            losses = torch.stack([(torch.sum(densities[j] - source_densities[j])**2/(2*len(densities[j]))) for j in range(num_domains)])
            domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                           F.nll_loss(tdomains[j], tlabels) for j in range(num_domains)])
            # Different final loss function depending on different training modes.
            if mode == "maxmin":
                loss = torch.max(losses) + mu * torch.min(domain_losses)
            elif mode == "dynamic":
                loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
            else:
                raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            logger.info("Iteration {}, loss = {}".format(t, running_loss))
    time_end = time.time()
    # Test on other domains.
    mdan.eval()
    target_insts = torch.tensor(target_insts, requires_grad=False).float().to(device)
    target_densities  = torch.tensor(target_densities).float()
    target_counts  = torch.tensor(target_counts).float()
    #preds_labels = torch.max(mdan.inference(target_insts), 1)[1].cpu().data.squeeze_()
    preds_densities, preds_counts = mdan.inference(target_insts)
    mse_density = torch.sum(preds_densities - target_densities)**2/preds_densities.shape[0]
    mse_count = torch.sum(preds_counts - target_counts)**2/preds_counts.shape[0]
    logger.info("{}:-\n\t Count MSE: {}, Density MSE: {} time used = {} seconds.".
                format(i, mse_count, mse_density, time_end - time_start))
    #results[data_name[i]] = pred_acc
