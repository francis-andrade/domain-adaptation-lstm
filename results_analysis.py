import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", help="Results filepath", type = str, default=".")
parser.add_argument("--parameter", help="Contains parameter", type=str, default="None")
parser.add_argument("--parameter2", help="Contains second parameter", type=str, default="None")
parser.add_argument('--make_graphics', help="Make grapfics", type=str, default=False)

args = parser.parse_args()

filepath = args.filepath

result_files = os.listdir(filepath)


def avg_dict(dict):
    sum = 0
    for k in dict.keys():
        sum += dict[k]
    return sum / len(dict.keys())

def verify_parameters(filename, parameter1, parameter2):
    return parameter1 == 'None' or (parameter1 in filename and (parameter2 == 'None' or parameter2 in filename))

res_total_mae = {}
avg_total_mae = {}
res_best_mae = {}
avg_best_mae = {}

for result_file in result_files:
    
    if verify_parameters(result_file, args.parameter, args.parameter2):
        keywords = result_file.split('_')
        if len(keywords) > 2:
            model = keywords[0]
            if model in ['simple', 'single', 'double', 'common']:
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