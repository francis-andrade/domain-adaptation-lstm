import settings
import argparse
import torch
import utils
import numpy as np
from model import MDANet
from model_temporal_common import MDANTemporalCommon
from model_temporal_double import MDANTemporalDouble
import torch.optim as optim
import torch.nn.functional as F
import pickle
from load_data import load_data, load_data_from_file, load_data_structure, load_insts ,CameraData, CameraTimeData, FrameData, VehicleData
import joblib
import gc
import plotter
import copy
import os



parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Name used to save the log file.", type = str, default="webcamT")

parser.add_argument("--seed", help="Random seed.", type=int, default=42)
parser.add_argument("--verbose", help="Verbose mode: True -- show training progress. False -- "
                                            "not show training progress.", type=bool, default=True)
parser.add_argument("--model", help="Choose a model to train: [mdan]",
                    type=str, default="mdan")

parser.add_argument("--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1e-2)
parser.add_argument('--lambda', default=1e-3, type=float, metavar='', help='trade-off between density estimation and vehicle count losses (see eq. 7 in the paper)')
parser.add_argument("--epochs", help="Number of training epochs", type=int, default=2)
parser.add_argument("--batch_size", help="Batch size during training", type=int, default=10)
parser.add_argument("--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="maxmin")
parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
parser.add_argument('--visdom_env', default='MDAN', type=str, metavar='', help='Visdom environment name')
parser.add_argument('--visdom_port', default=8444, type=int, metavar='', help='Visdom port')
parser.add_argument('--results_file', default='results', type=str, metavar='', help = 'Name for results file')
# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = utils.get_logger(args.name)
logger.info("Using device: "+str(device))
# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Loading the randomly partition the amazon data set.
logger.info('Started loading data')
#data = load_data(10)
#data = joblib.load('temporary.npy')

if settings.LOAD_MULTIPLE_FILES:
    data_insts, data_counts = load_insts('first', None, 50)
else:
    data_insts, data_densities, data_counts = load_insts('first', 'first', 50)

if settings.TEMPORAL:
    if settings.LOAD_MULTIPLE_FILES:
        data_insts, data_counts = utils.group_sequences_load_multiple_files(data_insts, data_counts, settings.SEQUENCE_SIZE)
    else:
        data_insts, data_densities, data_counts = utils.group_sequences(data_insts, data_densities, data_counts, settings.SEQUENCE_SIZE)

##############################
################################
num_epochs = args.epochs
batch_size = args.batch_size
num_domains = settings.NUM_DATASETS - 1
lr = 0.0001
mu = args.mu
gamma = 10.0
mode = args.mode
args_dict = vars(args)
lambda_ = args_dict["lambda"]

args_dict = vars(args)
if args_dict['use_visdom']:
    loss_plt = plotter.VisdomLossPlotter(env_name=args_dict['visdom_env'],
                                             port=args_dict['visdom_port'])
    img_plt = plotter.VisdomImgsPlotter(env_name=args_dict['visdom_env'],
                                            port=args_dict['visdom_port'])
logger.info("Training with domain adaptation using PyTorch madnNet: ")
error_dicts = {}
results = {}
results['count (mse)'] = {}
results['density (mse)'] = {}
results['count (mae)'] = {}
results['best count (mse)'] = {}
results['best density (mse)'] = {}
results['best count (mae)'] = {}
results['final count (mse)'] = {}
results['final density (mse)'] = {}
results['final count (mae)'] = {}

for i in range(settings.NUM_DATASETS):
    domain_id = settings.DATASETS[i]
    results['best density (mse)'][domain_id] = np.inf
    results['best count (mse)'][domain_id] = np.inf
    results['best count (mae)'][domain_id] = np.inf
    
    # Train DannNet.
    if settings.TEMPORAL:
        mdan = MDANTemporalDouble(num_domains, settings.IMAGE_NEW_SHAPE).to(device)
        #mdan = MDANTemporalCommon(num_domains, settings.IMAGE_NEW_SHAPE).to(device)
        best_mdan = MDANTemporalDouble(num_domains, settings.IMAGE_NEW_SHAPE).to(device)
    else:
        mdan = MDANet(num_domains).to(device)
        best_mdan = MDANet(num_domains).to(device)

    optimizer = optim.Adadelta(mdan.parameters(), lr=lr)

    size = len(data_insts[i])
    indices = np.random.permutation(size)
    test_idx = indices[int(0.7*size):]
    test_insts, test_counts = data_insts[i][test_idx], data_counts[i][test_idx]
    if not settings.LOAD_MULTIPLE_FILES:
        test_densities = data_densities[test_idx]
    val_idx = indices[:int(0.7*size)]
    val_insts, val_counts = data_insts[i][val_idx], data_counts[i][val_idx]
    if not settings.LOAD_MULTIPLE_FILES:
        val_densities = data_densities[i][val_idx]
    

    mdan.train()
    # Training phase.

    logger.info("Start training Domain: {}...".format(str(settings.DATASETS[i])))
    for t in range(num_epochs):
            running_loss = 0.0
            running_count_loss = 0.0
            running_density_loss = 0.0
            no_batches = 0
            if settings.LOAD_MULTIPLE_FILES:
                train_loader = utils.multi_data_loader(data_insts, None, data_counts, batch_size, 'first', 'proportional')
            else:
                train_loader = utils.multi_data_loader(data_insts, data_densities, data_counts, batch_size, 'first', 'proportional')
            for batch_insts, batch_densities, batch_counts in train_loader:
                #logger.info("Starting batch")
                # Build source instances.
                source_insts = []
                source_counts = []
                source_densities = []
                for j in range(settings.NUM_DATASETS):
                    if j != i:
                        source_insts.append(torch.from_numpy(np.array(batch_insts[j], dtype=np.float)).float().to(device))
                        source_counts.append(torch.from_numpy(np.array(batch_counts[j],  dtype=np.float)).float().to(device))
                        densities = np.array(batch_densities[j], dtype=np.float)
                        if settings.TEMPORAL:
                            N, T, C, H, W = densities.shape 
                            densities = np.reshape(densities, (N*T, C, H, W))
                        source_densities.append(torch.from_numpy(densities).float().to(device))
                
                tinputs = torch.from_numpy(np.array(batch_insts[i], dtype=np.float)).float().to(device)       
                optimizer.zero_grad()

                slabels = []
                tlabels = []
                for k in range(num_domains):
                    slabels.append(torch.ones(len(source_insts[k]), requires_grad=False).type(torch.LongTensor).to(device))
                    tlabels.append(torch.zeros(len(source_insts[k]), requires_grad=False).type(torch.LongTensor).to(device))
                #print("Starting MDAN")
                model_densities, model_counts, sdomains, tdomains = mdan(source_insts, tinputs)
                # Compute prediction accuracy on multiple training sources.
                density_losses = torch.stack([(torch.sum((model_densities[j] - source_densities[j])**2)/(len(model_densities[j]))) for j in range(num_domains)])
                count_losses = torch.stack([(torch.sum((model_counts[j] - source_counts[j])**2)/(len(model_counts[j]))) for j in range(num_domains)])
                losses = density_losses + lambda_*count_losses
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels[j]) +
                                           F.nll_loss(tdomains[j], tlabels[j]) for j in range(num_domains)])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) + mu * torch.min(domain_losses)
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                else:
                    raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
                no_batches += 1
                running_loss += loss.item()
                running_count_loss += count_losses.mean().item()
                running_density_loss += density_losses.mean().item()
                loss.backward()
                optimizer.step()
            
            logger.info("Iteration {}, loss = {}, mean count loss = {}, mean density loss = {}".format(t, running_loss, running_count_loss / no_batches, running_density_loss / no_batches))

            if args_dict['use_visdom']:
                # plot the losses
                loss_plt.plot('global loss ('+str(settings.DATASETS[i])+')', 'train', 'MSE', t, running_loss)
                loss_plt.plot('density loss ('+str(settings.DATASETS[i])+')', 'train', 'MSE', t, running_density_loss / no_batches)
                loss_plt.plot('count loss ('+str(settings.DATASETS[i])+')', 'train', 'MSE', t, running_count_loss / no_batches)

            if settings.LOAD_MULTIPLE_FILES:
                densities = None
            else:
                densities = val_densities
            mse_density, mse_count, mae_count = utils.eval_mdan(mdan, val_insts, densities, val_counts, batch_size, device)

            logger.info("Validation, Count MSE: {}, Density MSE: {}, Count MAE: {}".
                      format(mse_count, mse_density, mae_count))
           
            if args_dict['use_visdom']:
                    # plot the losses
                    loss_plt.plot('count error ('+str(domain_id)+')', 'valid', 'MAE', t, mae_count)
                    loss_plt.plot('density loss ('+str(domain_id)+')', 'valid', 'MSE', t, mse_density)
                    loss_plt.plot('count loss ('+str(domain_id)+')', 'valid', 'MSE', t, mse_count)

            if mse_density < results['best density (mse)'][domain_id]:
                    results['best density (mse)'][domain_id] = mse_density

            if mse_count < results['best count (mse)'][domain_id]:
                    results['best count (mse)'][domain_id] = mse_count
                    best_mdan.load_state_dict(mdan.state_dict())

            if mae_count < results['best count (mae)'][domain_id]:
                    results['best count (mae)'][domain_id] = mae_count
    
    results['density (mse)'][domain_id] = mse_density
    results['count (mse)'][domain_id] = mse_count
    results['count (mae)'][domain_id] = mae_count

    if settings.LOAD_MULTIPLE_FILES:
        densities = None
    else:
        densities = test_densities
    final_mse_density, final_mse_count, final_mae_count = utils.eval_mdan(best_mdan, test_insts, densities,test_counts, batch_size, device)

    results['final density (mse)'][domain_id] = final_mse_density
    results['final count (mse)'][domain_id] = final_mse_count
    results['final count (mae)'][domain_id] = final_mae_count
    
    logger.info("Testing, Count MSE: {}, Density MSE: {}, Count MAE: {}".
                      format(final_mse_count, final_mse_density, final_mae_count))
    
    #del train_loader, mdan, optimizer, source_insts, source_counts, source_densities, tinputs, target_insts, target_counts, target_densities, preds_densities, preds_counts, model_densities, model_counts, sdomains, tdomains, loss, domain_losses, slabels, tlabels
    #n_obj = gc.collect()
    #print('No. of objects removed: ', n_obj)





logger.info("Prediction accuracy with multiple source domain adaptation using madnNet: ")
logger.info(results)
pickle.dump(results, open(os.path.join(settings.DATASET_DIRECTORY, '../'+args.results_file+'.npy'), 'wb+'))