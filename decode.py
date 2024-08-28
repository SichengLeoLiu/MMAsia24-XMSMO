""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join, exists
from datetime import timedelta
from time import time
import pickle as pkl
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op
from torch.autograd import Variable
import numpy as np
import torch
from torch.utils.data import DataLoader

from torch import multiprocessing as mp, nn
#from utils import make_vocab
#from data.batcher import tokenize
# from clipsumcomp_topic_trans_3stream_modal_specific import CLIPSum
from clipsum import CLIPSum
from datasets import load_dataset
from util_dataset import EXMSMODataset
import cv2

try:
    DATA_DIR = os.environ['DATA']

except KeyError:
    print('please use environment variable to specify data directories')

def collate_func(inps):
    return [a for a in inps]

def decode(params, dataset_folder, save_path, model_dir, model_name, split, batch_size, cuda):
    start = time()

    summarizer = CLIPSum(**params)
    print("-----------summarizer")
    #print("summarizer", summarizer)
    summarizer.cuda()


    summarizer.load_state_dict(torch.load(join(model_dir,model_name)))
    summarizer.eval()

    dataset = EXMSMODataset('test', dataset_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    n_data = len(dataset)

    # prepare save paths and logs
    if not exists(join(save_path, 'outputText')):
        os.makedirs(join(save_path, 'outputText'))

    if not exists(join(save_path, 'outputTextExtIdx')):
        os.makedirs(join(save_path, 'outputTextExtIdx'))

    if not exists(join(save_path, 'outputPic')):
        os.makedirs(join(save_path, 'outputPic'))

    if not exists(join(save_path, 'outputPicExtIdx')):
        os.makedirs(join(save_path, 'outputPicExtIdx'))

    print("loader length", len(dataloader))
    # Decoding
    i = 0
    start_process=False
    if args.last_process == "":
        start_process=True

    print("start_process", start_process)
    with torch.no_grad():
        for ib, d in enumerate(dataloader):


            file_id = d[0][0]

            if  args.last_process == file_id:
                start_process=True
            if start_process:
                descriptions = d[1]
                videos = d[2]
                #scenes = d[6]
                #titles.append(d[3])
                #transcripts.append(d[4])
                print("file_id", file_id)
                print("descriptions", descriptions)
                print("videos", videos.size())
                # Forward pass
                #bodies = [doc[args.dataset_doc_field] for doc in documents]
                output_text_summaries, output_text_summaries_pos, text_logp, output_video_summaries, output_video_summaries_pos, video_logp = summarizer(descriptions, videos)

                    #sampled_summaries, _, sampled_pointers = summarizer.forward(bodies, args.max_ext_output_length, args.max_comp_output_length)

                
                #print("decoded", comp_sampled_summaries[0])

                #print("decoded ext_arts_w", ext_arts_w)
                with open(join(save_path, 'outputText/{}.dec'.format(file_id)),'w') as f:
                    f.write(output_text_summaries[0])

                with open(join(save_path, 'outputTextExtIdx/{}.dec'.format(file_id)),'w') as f:
                    f.write(','.join([str(sent) for sent in output_text_summaries_pos[0]] ))

                #print("videos[pics[0]]", videos[pics[0]].numpy().shape)
                #cv2.imwrite(join(save_path, 'outputPic/{}.png'.format(file_id)), cv2.cvtColor(output_video_summaries[0].numpy(), cv2.COLOR_BGR2RGB))
                cv2.imwrite(join(save_path, 'outputPic/{}.png'.format(file_id)), output_video_summaries[0].numpy())

                #with open(join(args.save_path, 'outputPic/{}.dec'.format(ib)),'w') as f:
                #    f.write(videos[pics[0]])
                with open(join(save_path, 'outputPicExtIdx/{}.dec'.format(file_id)),'w') as f:
                    f.write(','.join([str(pic) for pic in output_video_summaries_pos] )) #adjust position

                i += 1
    print()




if __name__ == '__main__':
    print("running")
    parser = argparse.ArgumentParser()


    parser.add_argument('--max_sequence_length', type=int, default=900)
    parser.add_argument('--max_article_length', type=int, default=5)
    parser.add_argument('--max_summary_pic', type=int, default=1)
    parser.add_argument('--max_summary_word', type=int, default=12)

    parser.add_argument('--test', action='store_true')
    parser.add_argument('-bs', '--batch', type=int, default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00005)

    parser.add_argument('-ths', '--text_hidden_size', type=int, default=128)
    parser.add_argument('-vhs', '--video_hidden_size', type=int, default=128)
    #parser.add_argument('-nah', '--num_attention_head', type=int, default=2)


    #parser.add_argument('-chs', '--conductor_hidden_size', type=int, default=256)
    #parser.add_argument('-dhs', '--decoders_hidden_size', type=int, default=64)
    #parser.add_argument('-dis', '--decoders_initial_size', type=int, default=32)

    #parser.add_argument('-nl', '--num_layers', type=int, default=2)

    parser.add_argument('--path', required=True, help='path to ext model')
    parser.add_argument('--model_dir', required=True, help='path to ext model')
    parser.add_argument('--model_name', required=True, help='ext model')
    parser.add_argument('--dataset_folder', type=str, help='folder of dataset')
    parser.add_argument('--last_process', type=str, default='')
    args = parser.parse_args()

    #args.cuda = torch.cuda.is_available() and not args.no_cuda
    print("torch.cuda.is_available()", torch.cuda.is_available())
    args.cuda = True

    params = dict(
        max_summary_word=args.max_summary_word,
        max_summary_pic=args.max_summary_pic,
        text_hidden_size=args.text_hidden_size,
        video_hidden_size=args.video_hidden_size,
        #num_attention_head=args.num_attention_head,
        #num_layers=args.num_layers,
    )
    data_split = 'test' if args.test else 'val'
    decode(params, args.dataset_folder, args.path, args.model_dir, args.model_name,
           data_split, args.batch, 
           args.cuda)
