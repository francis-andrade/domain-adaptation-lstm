"""
Module used to make an analysis of the data collected and draw graphics.
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from decimal import Decimal

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", help="Results filepath", type = str, default=".")
parser.add_argument("--parameter", help="Contains parameter", type=str, default="None")
parser.add_argument("--parameter2", help="Contains second parameter", type=str, default="None")
parser.add_argument('--make_graphics', help="Make grapfics", type=int, metavar='', default=False)
parser.add_argument('--unsupervised_filename', help="Filename for the unsupervised graph", type=str, metavar='', default='unsupervised')
parser.add_argument('--semisupervised_filename', help="Filename for the semisupervised graph", type=str, metavar='', default='semisupervised')

args = parser.parse_args()

filepath = args.filepath

result_files = os.listdir(filepath)


def avg_dict(dict):
    sum = 0
    for k in dict.keys():
        sum += dict[k]
    return sum / len(dict.keys())

def min_dict(dict):
    min_value = np.inf
    min_value_key = None

    for k in dict.keys():
        if dict[k] < min_value:
            min_value = dict[k]
            min_value_key = k
    
    return [min_value_key, min_value]

def verify_parameters(filename, parameter1, parameter2='None'):
    return parameter1 == 'None' or (parameter1 in filename and (parameter2 == 'None' or parameter2 in filename))

res_total_mae = {}
avg_total_mae = {}
res_best_mae = {}
avg_best_mae = {}

domain_unsupervised_best_mae = {}
domain_semisupervised_best_mae = {}
domain_unsupervised_avg_mae = {}
domain_semisupervised_avg_mae = {}

MODELS_DA = ['simple', 'single', 'double', 'common']
MODELS_ORIGINAL = ['original', 'original_temporal']
MODES = ['average', 'maxmin', 'dynamic']

for result_file in result_files:
    
    if verify_parameters(result_file, args.parameter, args.parameter2):
        keywords = result_file.split('_')
        if len(keywords) > 2:
            model = keywords[0]
            if model in MODELS_DA:
                mu = keywords[-1][:-4]
                if model not in res_total_mae:
                    res_total_mae[model] = {}
                    avg_total_mae[model] = {}
                    res_best_mae[model] = {}
                    avg_best_mae[model] = {}

                result_path = os.path.join(filepath, result_file)
                results = pickle.load(open(result_path, 'rb'))
                res_total_mae[model][float(mu)] = results['total count (mae)']
                avg_total_mae[model][float(mu)] = avg_dict(results['total count (mae)'])
                res_best_mae[model][float(mu)] = results['best val count (mae)']
                avg_best_mae[model][float(mu)] = avg_dict(results['best val count (mae)'])

            elif model in ['original']:
                if keywords[1] == 'temporal':
                    model = 'original_temporal'
                result_path = os.path.join(filepath, result_file)
                results = pickle.load(open(result_path, 'rb'))
                res_total_mae[model] = results['total count (mae)']
                avg_total_mae[model] = avg_dict(results['total count (mae)'])
                res_best_mae[model] = results['best val count (mae)']
                avg_best_mae[model] = avg_dict(results['best val count (mae)'])

modes = {}
for result_file in result_files:
    if verify_parameters(result_file, args.parameter, args.parameter2):
        keywords = result_file.split('_')
        if len(keywords) > 2:
            model = keywords[0]
            if model not in modes:
                modes[model] = {}
            if model in MODELS_DA:
                for mode in MODES:
                    if verify_parameters(result_file, mode):
                        result_path = os.path.join(filepath, result_file)
                        results = pickle.load(open(result_path, 'rb'))
                        modes[model][mode] = avg_dict(results['total count (mae)'])

for model in res_total_mae:
    domain_unsupervised_best_mae[model] = {}
    domain_unsupervised_avg_mae[model] = {}
    
    
    if model in MODELS_DA:
        domain_to_mu_dict = {}
        domain_to_mu_dict['avg'] = {}
        
        for mu in res_total_mae[model]:
            for domain in res_total_mae[model][mu]:
                if domain not in domain_to_mu_dict:
                    domain_to_mu_dict[domain] ={}
                domain_to_mu_dict[domain][mu] = round(res_total_mae[model][mu][domain], 3)
                domain_to_mu_dict['avg'][mu] = round(avg_total_mae[model][mu],3)

        for domain in domain_to_mu_dict:
            domain_unsupervised_avg_mae[model][domain] = round(avg_dict(domain_to_mu_dict[domain]), 3)
            domain_unsupervised_best_mae[model][domain] = min_dict(domain_to_mu_dict[domain])
    elif model in MODELS_ORIGINAL:
        domain_unsupervised_best_mae[model]['avg'] = round(avg_total_mae[model], 3)
        domain_unsupervised_avg_mae[model]['avg'] = round(avg_total_mae[model], 3)

        for domain in res_total_mae[model]:
            domain_unsupervised_best_mae[model][domain] = round(res_total_mae[model][domain], 3)
            domain_unsupervised_avg_mae[model][domain] = round(res_total_mae[model][domain], 3)
        

for model in res_best_mae:
    domain_semisupervised_best_mae[model] = {}
    domain_semisupervised_avg_mae[model] = {}

    

    if model in MODELS_DA:
        domain_to_mu_dict = {}
        domain_to_mu_dict['avg'] = {}

        for mu in res_best_mae[model]:
            for domain in res_best_mae[model][mu]:
                if domain not in domain_to_mu_dict:
                    domain_to_mu_dict[domain] ={}
                domain_to_mu_dict[domain][mu] = round(res_best_mae[model][mu][domain],3)
                domain_to_mu_dict['avg'][mu] = round(avg_best_mae[model][mu], 3)
        for domain in domain_to_mu_dict:
            domain_semisupervised_avg_mae[model][domain] = round(avg_dict(domain_to_mu_dict[domain]), 3)
            domain_semisupervised_best_mae[model][domain] = min_dict(domain_to_mu_dict[domain])
    elif model in MODELS_ORIGINAL:
        domain_semisupervised_avg_mae[model]['avg'] = round(avg_best_mae[model], 3)
        domain_semisupervised_best_mae[model]['avg'] = round(avg_best_mae[model], 3)

        for domain in res_best_mae[model]:
            domain_semisupervised_avg_mae[model][domain] = round(res_best_mae[model][domain], 3)
            domain_semisupervised_best_mae[model][domain] = round(res_best_mae[model][domain], 3)
        


def format_e(n):
    a = '%E' % Decimal(str(n))
    return a.split('E')[0].rstrip('0').rstrip('.') + 'e-' + a.split('-')[1].lstrip('0')

def rgb(colors):
    return tuple(np.array(colors)/255.0)

GRAPHICS_MODELS = ['simple', 'single', 'double', 'common']
GRAPHICS_COLORS = [rgb((83,81,84)), rgb((218, 124, 48)), rgb((57,106,177)), rgb((146,36,40))]

def make_plot(ylabel, dict, filename):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    mus = list(dict['simple'].keys())
    mus.sort()
    mus_str = [format_e(mu) for mu in mus]
    print(mus)
    values = []
    for model_idx in range(len(GRAPHICS_MODELS)):
        model = GRAPHICS_MODELS[model_idx]
        values = []
        for key in mus:
            values.append(dict[model][key])
        ax.plot(mus_str, values, color=GRAPHICS_COLORS[model_idx], label=GRAPHICS_MODELS[model_idx])
    #ax.set_xticklabels(mus_str)
    ax.set_xlabel(r'$\mathbf{\mu}$', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height*0.8])
    leg = ax.legend(loc='center left', frameon=True, bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True)
    #plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    

if args.make_graphics:
    unsupervised_filename = args.unsupervised_filename + '.png'
    make_plot('Average MAE Count (Unsupervised)', avg_total_mae, unsupervised_filename)
    semisupervised_filename = args.semisupervised_filename + '.png'
    make_plot('Average MAE Count (Semisupervised)', avg_best_mae, semisupervised_filename)

def order_results_ucspeds(model_name):
    return [domain_semisupervised_best_mae[model_name]['vidd'], domain_semisupervised_best_mae[model_name]['vidf'], domain_semisupervised_best_mae[model_name]['avg']]

def order_results_webcamt(model_name):
    dict = domain_semisupervised_best_mae[model_name]
    return [dict[511], dict[551], dict[691], dict[846], dict['avg']]

