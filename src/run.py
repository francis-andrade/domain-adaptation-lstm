import settings
import argparse
import torch
import utils
import numpy as np
from model import MDANet
from model_temporal_common import MDANTemporalCommon
from model_temporal_double import MDANTemporalDouble
from model_temporal_single import MDANTemporalSingle
import torch.optim as optim
import torch.nn.functional as F
import pickle
from load_webcamt import CameraData, CameraTimeData, FrameData, VehicleData
import load_webcamt
from load_ucspeds import VideoDataUCS, FrameDataUCS
import load_ucspeds
import joblib
import gc
import plotter
import copy
import os



parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Name used to save the log file.", type = str, default="webcamT")

parser.add_argument("--seed", help="Random seed.", type=int, default=42)

parser.add_argument("--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1e-2)
parser.add_argument('--lambda', default=1e-2, type=float, metavar='', help='trade-off between density estimation and vehicle count losses (see eq. 7 in the paper)')
parser.add_argument("--epochs", help="Number of training epochs", type=int, default=2)
parser.add_argument("--batch_size", help="Batch size during training", type=int, default=10)
parser.add_argument("--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic|average]", type=str, default="maxmin")
parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
parser.add_argument('--visdom_env', default='MDAN', type=str, metavar='', help='Visdom environment name')
parser.add_argument('--visdom_port', default=8444, type=int, metavar='', help='Visdom port')
parser.add_argument('--results_file', default='None', type=str, metavar='', help = 'Name for results file')
parser.add_argument('--cuda', default='0', type=str, metavar='', help = 'CUDA GPU')
parser.add_argument('--lr', default=1e-3, type=float, metavar='', help='Learning Rate')
parser.add_argument('--model', default='simple', type=str, metavar='', help='Model [simple|common|double|single]')
parser.add_argument('--prefix_data', default='first', type=str, metavar='', help='Data Prefix')
parser.add_argument('--prefix_densities', default='first', type=str, metavar='', help='Densities Prefix')
parser.add_argument('--dataset', default='webcamt', type=str, metavar='', help='Dataset [webcamt|ucspeds]')
parser.add_argument('--use_mask', default=True, type=int, metavar='', help='Use mask')
parser.add_argument('--use_transformations', default=True, type=int, metavar='', help='Use Data Augmentation')
# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")

logger = utils.get_logger(args.name)
logger.info("Using device: "+str(device))
# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Loading the randomly partition the amazon data set.
logger.info('Started loading data')

settings.DATASET = args.dataset

if args.model == 'simple':
    settings.TEMPORAL = False
else:
    settings.TEMPORAL = True

settings.PREFIX_DENSITIES = args.prefix_densities
settings.PREFIX_DATA = args.prefix_data

settings.USE_MASK = args.use_mask

if args.use_transformations:
    transforms = settings.TRANSFORMS
else:
    transforms = []

if args.dataset == 'webcamt':
    data, data_insts, data_counts = load_webcamt.load_insts(settings.PREFIX_DATA, 20)
elif args.dataset == 'ucspeds':
    data, data_insts, data_counts = load_ucspeds.load_insts(settings.PREFIX_DATA, 20)

if settings.TEMPORAL:
    data_insts, data_counts = utils.group_sequences(data_insts, data_counts, settings.SEQUENCE_SIZE)


##############################
################################
num_epochs = args.epochs
batch_size = args.batch_size
num_domains = len(data_insts) - 1
lr = args.lr
mu = args.mu
gamma = 10.0
mode = args.mode
args_dict = vars(args)
lambda_ = args_dict["lambda"]

if args.results_file == "None":
    results_file = args.model+'_'+settings.PREFIX_DENSITIES+'_'+args.mode+'_'+str(args.lr)+'_'+str(args.mu)
else:
    results_file = args.results_file

if args_dict['use_visdom']:
    loss_plt = plotter.VisdomLossPlotter(env_name=args.dataset+'_'+results_file,
                                             port=args_dict['visdom_port'])

logger.info("Training with domain adaptation using PyTorch madnNet: ")
error_dicts = {}
results = {}
results['total count (mse)'] = {}
results['total density (mse)'] = {}
results['total count (mae)'] = {}
results['best val count (mse)'] = {}
results['best val density (mse)'] = {}
results['best val count (mae)'] = {}
results['test count (mse)'] = {}
results['test density (mse)'] = {}
results['test count (mae)'] = {}

counts_register = []
for i in range(len(data_insts)):
    counts_register.append([])

for i in range(len(data_insts)):
    if args.dataset == 'webcamt':
        domain_id = settings.WEBCAMT_DOMAINS[i]
    else:
        domain_id = settings.UCSPEDS_DOMAINS[i]
    results['best val density (mse)'][domain_id] = np.inf
    results['best val count (mse)'][domain_id] = np.inf
    results['best val count (mae)'][domain_id] = np.inf
    
    # Train DannNet.
    if args.model == 'simple':
        mdan = MDANet(num_domains).to(device)
        best_mdan = MDANet(num_domains).to(device)
    elif args.model == 'common':
        mdan = MDANTemporalCommon(num_domains, settings.get_new_shape()).to(device)
        best_mdan = MDANTemporalCommon(num_domains, settings.get_new_shape()).to(device)
    elif args.model == 'double':
        mdan = MDANTemporalDouble(num_domains, settings.get_new_shape()).to(device)
        best_mdan = MDANTemporalDouble(num_domains, settings.get_new_shape()).to(device)
    elif args.model == 'single':
        mdan = MDANTemporalSingle(num_domains, settings.get_new_shape()).to(device)
        best_mdan = MDANTemporalSingle(num_domains, settings.get_new_shape()).to(device)

    optimizer = torch.optim.Adam(mdan.parameters(), lr=lr, weight_decay=0)

    size = len(data_insts[i])
    indices = np.random.permutation(size)
    test_idx = indices[int(settings.VALIDATION_TEST_RATIO*size):]
    test_insts, test_counts = data_insts[i][test_idx], data_counts[i][test_idx]
    val_idx = indices[:int(settings.VALIDATION_TEST_RATIO*size)]
    val_insts, val_counts = data_insts[i][val_idx], data_counts[i][val_idx]           

    
    # Training phase.

    logger.info("Start training Domain: {}...".format(str(domain_id)))
    for t in range(num_epochs):
            mdan.train()
            running_loss = 0.0
            running_count_loss = 0.0
            running_density_loss = 0.0
            no_batches = 0
            train_loader = utils.multi_data_loader(data_insts, data_counts, batch_size, settings.PREFIX_DATA, settings.PREFIX_DENSITIES, data, transforms=transforms)
            
            for batch_insts, batch_densities, batch_counts, batch_masks in train_loader:
                #logger.info("Starting batch")
                # Build source instances.
                source_insts = []
                source_counts = []
                source_densities = []
                if settings.USE_MASK:
                    source_masks = []
                else:
                    source_masks = None
                for j in range(len(data_insts)):
                    if j != i:
                        source_insts.append(torch.from_numpy(np.array(batch_insts[j], dtype=np.float) / 255).float().to(device))  
                        
                        densities = np.array(batch_densities[j], dtype=np.float)
                        if settings.TEMPORAL:
                            N, T, C, H, W = densities.shape 
                            densities = np.reshape(densities, (N*T, C, H, W))
                        source_densities.append(torch.from_numpy(densities).float().to(device))
                        if settings.USE_MASK:
                            masks = np.array(batch_masks[j],  dtype=np.float)
                            if settings.TEMPORAL:
                                N, T, C, H, W = masks.shape 
                                masks = np.reshape(masks, (N*T, C, H, W))
                            source_masks.append(torch.from_numpy(masks).float().to(device))
                        if settings.USE_MASK:
                            source_counts.append(torch.sum(source_densities[-1]*source_masks[-1], dim=(1,2,3)).reshape(N,T))
                        else:
                            source_counts.append(torch.from_numpy(np.array(batch_counts[j],  dtype=np.float)).float().to(device))
                if t==0:
                    counts_register[i].append(source_counts)
                
                tinputs = torch.from_numpy(np.array(batch_insts[i], dtype=np.float) / 255.0).float().to(device)   
                if settings.USE_MASK:
                    tmask = torch.from_numpy(np.array(batch_masks[i], dtype=np.float)).float().to(device)   
                else:
                    tmask = None
                optimizer.zero_grad()

                slabels = []
                tlabels = []
                for k in range(num_domains):
                    slabels.append(torch.ones(len(source_insts[k]), requires_grad=False).type(torch.LongTensor).to(device))
                    tlabels.append(torch.zeros(len(source_insts[k]), requires_grad=False).type(torch.LongTensor).to(device))
                model_densities, model_counts, sdomains, tdomains = mdan(source_insts, tinputs, source_masks, tmask)
                # Compute prediction accuracy on multiple training sources.
                density_losses = torch.stack([(torch.sum((model_densities[j] - source_densities[j])**2)/(len(model_densities[j]))) for j in range(num_domains)])
                count_losses = torch.stack([(torch.sum((model_counts[j] - source_counts[j])**2)/(len(model_densities[j]))) for j in range(num_domains)])
                losses = density_losses + lambda_*count_losses
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels[j]) +
                                           F.nll_loss(tdomains[j], tlabels[j]) for j in range(num_domains)])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) + mu * torch.min(domain_losses)
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                elif mode == 'average':
                    loss = torch.mean(losses+mu*domain_losses)
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
                loss_plt.plot('global loss ('+str(domain_id)+')', 'train', 'MSE', t, running_loss)
                loss_plt.plot('density loss ('+str(domain_id)+')', 'train', 'MSE', t, running_density_loss / no_batches)
                loss_plt.plot('count loss ('+str(domain_id)+')', 'train', 'MSE', t, running_count_loss / no_batches)

           
           
            mse_density, mse_count, mae_count = utils.eval_mdan(mdan, val_insts,  val_counts, batch_size, device, settings.PREFIX_DATA, settings.PREFIX_DENSITIES, data)

            logger.info("Validation, Count MSE: {}, Density MSE: {}, Count MAE: {}".
                      format(mse_count, mse_density, mae_count))
           
            if args_dict['use_visdom']:
                    # plot the losses
                    loss_plt.plot('count error ('+str(domain_id)+')', 'valid', 'MAE', t, mae_count)
                    loss_plt.plot('density loss ('+str(domain_id)+')', 'valid', 'MSE', t, mse_density)
                    loss_plt.plot('count loss ('+str(domain_id)+')', 'valid', 'MSE', t, mse_count)

            if mse_density < results['best val density (mse)'][domain_id]:
                    results['best val density (mse)'][domain_id] = mse_density

            if mse_count < results['best val count (mse)'][domain_id]:
                    results['best val count (mse)'][domain_id] = mse_count
                    best_mdan.load_state_dict(mdan.state_dict())

            if mae_count < results['best val count (mae)'][domain_id]:
                    results['best val count (mae)'][domain_id] = mae_count
    
    total_mse_density, total_mse_count, total_mae_count = utils.eval_mdan(mdan, data_insts[i],  data_counts[i], batch_size, device, settings.PREFIX_DATA, settings.PREFIX_DENSITIES, data)

    results['total density (mse)'][domain_id] = total_mse_density
    results['total count (mse)'][domain_id] = total_mse_count
    results['total count (mae)'][domain_id] = total_mae_count

    logger.info("All Target Data, Count MSE: {}, Density MSE: {}, Count MAE: {}".
                      format(total_mse_count, total_mse_density, total_mae_count))

    final_mse_density, final_mse_count, final_mae_count = utils.eval_mdan(best_mdan, test_insts, test_counts, batch_size, device, settings.PREFIX_DATA, settings.PREFIX_DENSITIES, data)

    results['test density (mse)'][domain_id] = final_mse_density
    results['test count (mse)'][domain_id] = final_mse_count
    results['test count (mae)'][domain_id] = final_mae_count
    
    logger.info("Testing, Count MSE: {}, Density MSE: {}, Count MAE: {}".
                      format(final_mse_count, final_mse_density, final_mae_count))



logger.info("Prediction accuracy with multiple source domain adaptation using madnNet: ")
results['args'] = args_dict
logger.info(results)

pickle.dump(results, open(os.path.join(settings.DATASET_DIRECTORY, '../Results/'+args.dataset+'/'+results_file+'.npy'), 'wb+'))