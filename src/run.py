import settings
import time
import argparse
import torch
import utils
import numpy as np
from model import MDANet
from model_temporal import MDANTemporal
import torch.optim as optim
import torch.nn.functional as F
import pickle
from load_data import load_data, load_data_densities, CameraData, CameraTimeData, FrameData, VehicleData
import joblib
import gc

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type = str, default="webcamT")

parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                            "not show training progress.", type=bool, default=True)
parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                    type=str, default="mdan")

parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1e-2)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=1)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=10)
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
logger.info('Started loading data')
#data = load_data(10)
#data = joblib.load('temporary.npy')
data =  load_data_densities('first', 'first')
logger.info('Finished loading data')
data_insts, data_densities, data_counts = [], [], []

for domain_id in data:
    domain_insts, domain_densities, domain_counts = [], [], []
    
    new_num_insts = 0
    for time_id in data[domain_id].camera_times:
        if new_num_insts > 50:
            break
        new_data_insts, new_data_densities, new_data_counts = [], [], []
        frame_ids = list(data[domain_id].camera_times[time_id].frames.keys())
        frame_ids.sort()
        for frame_id in frame_ids:
            if new_num_insts > 50:
                break
            if data[domain_id].camera_times[time_id].frames[frame_id].frame is not None:
                new_data_insts.append(data[domain_id].camera_times[time_id].frames[frame_id].frame / 255)
                new_data_densities.append(data[domain_id].camera_times[time_id].frames[frame_id].density)
                new_data_counts.append(len(data[domain_id].camera_times[time_id].frames[frame_id].vehicles))
                new_num_insts += 1
            else:
                print('None')
        
        if settings.TEMPORAL:
            domain_insts.append(new_data_insts)
            domain_densities.append(new_data_densities)
            domain_counts.append(new_data_counts)
        else:
            domain_insts += new_data_insts
            domain_densities += new_data_densities
            domain_counts += new_data_counts

    data_insts.append(np.array(domain_insts))
    data_counts.append(np.array(domain_counts))
    data_densities.append(np.array(domain_densities))

del data
n_obj = gc.collect()
print('Objects removed: ', n_obj)
if settings.TEMPORAL:
    data_insts, data_densities, data_counts = utils.group_sequences(data_insts, data_densities, data_counts, settings.SEQUENCE_SIZE)

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
results['count'] = {}
results['density'] = {}

for i in range(settings.NUM_DATASETS):
    
    # Train DannNet.
    if settings.TEMPORAL:
        mdan = MDANTemporal(num_domains, settings.IMAGE_NEW_SHAPE).to(device)
    else:
        mdan = MDANet(num_domains).to(device)
    optimizer = optim.Adadelta(mdan.parameters(), lr=lr)
    mdan.train()
    # Training phase.
    time_start = time.time()
    logger.info("Start training...")
    for t in range(num_epochs):
            running_loss = 0.0
            train_loader = utils.multi_data_loader(data_insts, data_densities, data_counts, batch_size)
            for batch_insts, batch_densities, batch_counts in train_loader:
                logger.info("Starting batch")
                # Build source instances.
                source_insts = []
                source_counts = []
                source_densities = []
                for j in range(settings.NUM_DATASETS):
                    if j != i:
                        source_insts.append(torch.from_numpy(np.array(batch_insts[j], dtype=np.float)).float().to(device))
                        source_counts.append(torch.from_numpy(np.array(batch_counts[j],  dtype=np.float)).float().to(device))
                        density = np.array(batch_densities[j], dtype=np.float)
                        if settings.TEMPORAL:
                            N, T, C, H, W = density.shape 
                            density = np.reshape(density, (N*T, 1, C, H, W))
                        source_densities.append(torch.from_numpy(density).float().to(device))

                tinputs = torch.from_numpy(np.array(batch_insts[i], dtype=np.float)).float().to(device)       
                optimizer.zero_grad()

                slabels = []
                tlabels = []
                for k in range(num_domains):
                    slabels.append(torch.ones(len(source_insts[k]), requires_grad=False).type(torch.LongTensor).to(device))
                    tlabels.append(torch.zeros(len(source_insts[k]), requires_grad=False).type(torch.LongTensor).to(device))
                
                model_densities, model_counts, sdomains, tdomains = mdan(source_insts, tinputs)
                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([(torch.sum(model_densities[j] - source_densities[j])**2/(2*len(model_densities[j]))) for j in range(num_domains)])
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels[j]) +
                                           F.nll_loss(tdomains[j], tlabels[j]) for j in range(num_domains)])
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
    # Build target instances.
    target_idx = i
    target_insts = np.array(data_insts[i], dtype=np.float)
    target_densities = np.array(data_densities[i], dtype=np.float)
    target_counts = np.array(data_counts[i], dtype=np.float)
    
    with torch.no_grad():
        mdan.eval()
        target_insts = torch.tensor(target_insts, requires_grad=False).float().to(device)
        target_densities  = torch.tensor(target_densities).float()
        target_counts  = torch.tensor(target_counts).float()
        #preds_labels = torch.max(mdan.inference(target_insts), 1)[1].cpu().data.squeeze_()
        preds_densities, preds_counts = mdan.inference(target_insts)
        mse_density = torch.sum(preds_densities - target_densities)**2/preds_densities.shape[0]
        mse_count = torch.sum(preds_counts - target_counts)**2/preds_counts.shape[0]
        logger.info("Domain {}:-\n\t Count MSE: {}, Density MSE: {} time used = {} seconds.".
                format(i, mse_count, mse_density, time_end - time_start))
        results['density'][i] = mse_density
        results['count'][i] = mse_count
    
    del train_loader, mdan, optimizer, source_insts, source_counts, source_densities, tinputs, target_insts, target_counts, target_densities, preds_densities, preds_counts, model_densities, model_counts, sdomains, tdomains, loss, domain_losses, slabels, tlabels
    n_obj = gc.collect()
    print('No. of objects removed: ', n_obj)