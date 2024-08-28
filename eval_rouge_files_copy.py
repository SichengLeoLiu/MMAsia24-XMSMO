""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import os
from os.path import join, exists

from evaluate_text import eval_meteor, eval_rouge
from rouge_metric import PyRouge
import re
from pyrouge import Rouge155
import shutil 
import glob

def main(args):
    dec_dir = join(args.decode_dir, args.decode_folder)
    print("dec_dir", dec_dir)
    #dec_dir = join(args.decode_dir, 'output')
    #with open(join(args.decode_dir, 'log.json')) as f:
    #    split = json.loads(f.read())['split']
    split = 'test'
    ref_dir = join(args.ref_dir, split)
    # print(ref_dir)
    assert exists(ref_dir)
    print("ref_dir", ref_dir)
    outputs = dict()
    rouge_scores =[]
    if args.rouge:

        dec_pattern = r'([A-z0-9_-]+).txt'
        # dec_pattern = r'([A-z0-9_-]+).dec'
        ref_pattern = '[A-z0-9_-]+.ref'
        cur_dir = args.decode_dir + 'current_file' 
        print("_count_data(dec_dir)", _count_data(dec_dir))
        for i in _count_data(dec_dir):

            if exists(cur_dir):
                shutil.rmtree(cur_dir)
            os.mkdir(cur_dir)
            shutil.copyfile(join(dec_dir, str(i)), join(cur_dir, str(i)))

            output = eval_rouge(dec_pattern, cur_dir, ref_pattern, ref_dir)
            metric = 'rouge_file'
            outputs[i] = output
            print(str(i) + output)            
            rouge_scores.append(parse_rouge_output(output))

        average_rouge = calculate_average_rouge(rouge_scores)
        print("Average ROUGE: ", average_rouge)
        with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
            f.write(str(outputs))
    
    else:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'

        print(output)

        with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
            f.write(output)



def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'([A-z0-9_-]+).txt')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return list(filter(match, names))

def parse_rouge_output(output):
    """ Parse the ROUGE output and extract the recall scores for ROUGE-1, ROUGE-2, and ROUGE-L. """
    rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    for line in output.split('\n'):
        if 'ROUGE-1' in line and 'Average_R' in line:
            rouge_scores['rouge-1'].append(float(line.split()[3]))
        elif 'ROUGE-2' in line and 'Average_R' in line:
            rouge_scores['rouge-2'].append(float(line.split()[3]))
        elif 'ROUGE-L' in line and 'Average_R' in line:
            rouge_scores['rouge-l'].append(float(line.split()[3]))
    return rouge_scores

def calculate_average_rouge(rouge_scores_list):
    """ Calculate the average ROUGE recall scores from a list of individual scores. """
    average_rouge = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
    count = len(rouge_scores_list)
    
    for scores in rouge_scores_list:
        for rouge_type in scores:
            average_rouge[rouge_type] += sum(scores[rouge_type]) / len(scores[rouge_type])

    for rouge_type in average_rouge:
        average_rouge[rouge_type] /= count
    
    return average_rouge

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the output files for the RL full models')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')
    parser.add_argument('--ref_dir', action='store', required=True,
                        help='directory of ref summaries')
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--decode_folder', action='store', required=True,
                        help='folder of decoded summaries')

    args = parser.parse_args()
    main(args)
