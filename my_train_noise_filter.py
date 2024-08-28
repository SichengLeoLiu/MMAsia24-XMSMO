import sys
#site_packages = next(p for p in sys.path if 'site-packages' in p)
#print(site_packages)
from PIL import Image
import numpy as np
import os, shutil
import re
import codecs
import os, pickle as pkl
from util_dataset import EXMSMODataset
#from MMVAE import MMVAE
#from clipsumcomp import CLIPSum
#from clipsumcomp_topic import CLIPSum
from my_model_noise_filter import CLIPSum
import codecs
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
from torch import save
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils import make_vocab, make_embedding, convert_word2id, to_var, idx2word
#from transformers import BertModel
from transformers import BertTokenizer
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer
from torchvision import transforms
import os
import torch.nn.functional as F
import torch.nn as nn
import gensim
import torch.nn.utils.rnn as rnn_utils
from multiprocessing import cpu_count
import time
import torch, gc
from collections import OrderedDict, defaultdict
from utils import PAD, UNK, END, START
import json
from torch.nn.functional import softplus
import torchvision.models as models
from datasets import load_dataset
import cv2 ,ot
import ssl
from model_generator import GeneTransformer
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
gc.collect()


ssl._create_default_https_context = ssl._create_unverified_context
BERT_NUM_TOKEN = 30522
torch.manual_seed(12345)



class TextCoverageLoss:
    # Depending on how many words are used a large fraction of the last X summaries
    def __init__(self, device="cuda", costmatrix_filename="COST_MATRIX_bert.pickle"):
    #def __init__(self, device="cpu", costmatrix_filename="COST_MATRIX.pickle"):
        
        #self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
        #self.model.eval()
        #self.tokenizer = utils_tokenizer.GPT2Tokenizer()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_bytes = 2**31 - 1

        bytes_in = bytearray(0)
        input_size = os.path.getsize(costmatrix_filename)
        with open(costmatrix_filename, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)

            self.COST_MATRIX = pkl.loads(bytes_in)
            #self.COST_MATRIX = pkl.load(f_in, map_location=torch.device('cpu'))
            #self.COST_MATRIX = torch.load(f_in, map_location=torch.device('cpu'))
        #self.COST_MATRIX = np.negative(self.COST_MATRIX)
        #self.COST_MATRIX = np.reciprocal(self.COST_MATRIX)

    def score(self, summaries, bodies):
        scores = []
         # Avoid changing p and q outside of this function
        with torch.no_grad():
            for i in range(len(summaries)):

                #doc = remove_stopwords(bodies[i])
                #summary = remove_stopwords(summaries[i])
                summary = summaries[i]
                doc = bodies[i]
                if len(summary)==0:
                    score = 1
                else:
                    
                    summary_token = self.tokenizer.encode(summary) 
                    body_token = self.tokenizer.encode(doc)

                    summary_bow = construct_BOW(summary_token)
                    body_bow = construct_BOW(body_token)

                    score = sparse_ot(summary_bow, body_bow, self.COST_MATRIX) 

                scores.append(score)
        
        print('text coverage score', scores)
        return sum(scores)/len(scores)


class MmCoverageLoss:
    def __init__(self, device="cuda"):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.featureextractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.clip.cuda()
        COSTMATRIX_DIM = 512
        #self.cosloss = nn.CosineEmbeddingLoss()
        self.COST_MATRIX = torch.ones(COSTMATRIX_DIM, COSTMATRIX_DIM) -  torch.eye(COSTMATRIX_DIM)
        self.COST_MATRIX = self.COST_MATRIX/COSTMATRIX_DIM


    def score(self, text_summaries, video_summaries, texts, videos):
        scores = []
        #for text, image in zip(text_summaries, video_text_summaries):
        #    print("text", text)
        #print("text_summaries", text_summaries)
        #print("video_text_summaries", video_summaries[0].size())
        with torch.no_grad():
            for v, t in zip(video_summaries, text_summaries):
                print("v", v.size())
                #print("t", t.size())
                i = transforms.ToPILImage()(v.permute(2,0,1).squeeze_(0))
                text_t = self.tokenizer(t, return_tensors = "pt", padding=True, truncation=True)
                image_t = self.featureextractor(i, return_tensors = "pt")
                print("image_t['pixel_values']", image_t['pixel_values'].size())
                text_f = self.clip.get_text_features(text_t['input_ids'].cuda())
                image_f = self.clip.get_image_features(image_t['pixel_values'].cuda())

                print("text_f", text_f.size())
                print("image_f", image_f.size())
                score = sparse_ot(text_f.squeeze(0).cpu().detach().numpy(), image_f.squeeze(0).cpu().detach().numpy(), self.COST_MATRIX.numpy()) 
                scores.append(score)
        #scores.append(self.cosloss(text_f, image_f, Variable(torch.ones(text_f.size()[0]).cuda())))
        #return scores[0]
        #return sum(scores)/len(scores)
        return sum(scores)/len(scores)

class MmAlignmentLoss:
    def __init__(self, device="cuda"):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.featureextractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        #self.cliptext = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        #self.clipvision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
       # self.cliptext.eval()
        #self.clipvision.eval()

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.clip.cuda()

        self.cosloss = nn.CosineEmbeddingLoss()

    def score(self, text_summaries, video_summaries):
        scores = []
        #for text, image in zip(text_summaries, video_text_summaries):
        #    print("text", text)
        #print("text_summaries", text_summaries)
        #print("video_text_summaries", video_summaries[0].size())
        text_t = self.tokenizer(text_summaries, return_tensors = "pt", padding=True, truncation=True)
        #print("text_t['input_ids']", text_t['input_ids'].size())
        image_batch = []
        for batch in video_summaries:
            image_batch.append(transforms.ToPILImage()(batch.permute(2,0,1).squeeze_(0)))

        image_t = self.featureextractor(image_batch, return_tensors = "pt")
        #print("image_t['pixel_values']", image_t['pixel_values'].size())
        text_f = self.clip.get_text_features(text_t['input_ids'].cuda())
        image_f = self.clip.get_image_features(image_t['pixel_values'].cuda())

        #print("text_f", text_f.size())
        #print("image_f", image_f.size())

        scores.append(self.cosloss(text_f, image_f, Variable(torch.ones(text_f.size()[0]).cuda())))



        return scores[0]

def VideoCoverageLoss(summaries, bodies):
    scores = []
    # Avoid changing p and q outside of this function
    
    for summary, video in zip(summaries, bodies):
        video = np.mean(np.array(video), axis=0).astype(np.float32)
        #summary = np.array(summary).reshape((128, 64)).astype(np.float32)
        summary = summary.detach().numpy().astype(np.float32)

        #video_64 = cv2.fromarray(video)
        #video_32 = cv2.cv.CreateMat(video.rows, video.cols, cv2.CV_32FC1)
        #video_32 = np.zeros((video.shape[0], video.shape[1], 1), dtype = np.float32)

        #cv2.Convert(video, video_32)

        #summary_64 = cv2.fromarray(summary)
        #summary_32 = np.zeros((summary_32.shape[0], summary_32.shape[1], 1), dtype = np.float32)

        #summary_32 = cv2.cv.CreateMat(summary.rows, summary.cols, cv2.CV_32FC1)
        #cv2.Convert(summary, summary_32)

        video_bw = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        summary_bw = cv2.cvtColor(summary, cv2.COLOR_BGR2GRAY)

        #print("video_bw", video_bw.shape)
        #print("summary_bw", summary_bw.shape)
        #print("video_bw", video_bw)
        #print("summary_bw", summary_bw)
        score = 1.0 / 0.001
        try:
        #black_image = cv2.cvtColor((np.ones((256,256,3))*255).astype(np.float32), cv2.COLOR_BGR2GRAY)
        #white_image = cv2.cvtColor((np.ones((256,256,3))*0.001).astype(np.float32), cv2.COLOR_BGR2GRAY)
    
        #scale = cv2.EMD(black_image,white_image,cv2.DIST_L2)[0]
        #print("scale", scale)
    #score = cv2.EMD(summary_bw,video_bw,cv2.DIST_L2)[0] / cv2.EMD(np.ones((500, 500, 1), dtype = "uint8")*0.001,np.ones((500, 500, 1), dtype = "uint8"),cv2.DIST_L2)[0]
            #score = cv2.EMD(summary_bw,video_bw,cv2.DIST_L2)[0] / scale
            score = cv2.EMD(summary_bw,video_bw,cv2.DIST_L2)[0]
        except:
            print("VideoCoverageLoss cannot compute")
        scores.append(score)
        
        ## change the latent representation to the actual video/image
    print('VideoCoverageLoss', scores)
    return sum(scores)/len(scores)

class OT_topic():
    def __init__(self):
        with open("kmeans_model.pkl", "rb") as f:
            self.k_means =  pkl.load(f)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.featureextractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        #self.cliptext = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        #self.clipvision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
       # self.cliptext.eval()
        #self.clipvision.eval()

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.clip.cuda()

        self.topic_size = 512

        self.COST_MATRIX = torch.ones(self.topic_size, self.topic_size) - torch.eye(self.topic_size)
        self.test_criterion = nn.MSELoss()


    def score(self, summaries_f, bodies_f):
        scores = []

        # topic_distance_summaries = self.k_means.transform(summaries_f)  # 确保仍然在张量上
        # topic_distance_bodies = self.k_means.transform(bodies_f)  # 确保仍然在张量上

        for summ, bod in zip(summaries_f, bodies_f):
            
            # score = sparse_ot(summ, bod, self.COST_MATRIX)  # 这里的 sparse_ot 需要是一个支持自动微分的 PyTorch 实现
            print("test loss7")
            score = self.test_criterion(summ, bod)
            scores.append(score)

        return torch.stack(scores)  # 使用 torch.stack 来合并得分列表成一个张量


    def score_text(self, summaries, bodies):
        
        with torch.no_grad():
            bodies_t = self.tokenizer(bodies, return_tensors = "pt", padding=True, truncation=True)
            # summaries_t = self.tokenizer(summaries, return_tensors = "pt", padding=True, truncation=True)
            
            bodies_f = self.clip.get_text_features(bodies_t['input_ids'].cuda())
            # summaries_f = self.clip.get_text_features(summaries_t['input_ids'].cuda())
        # print("text summaries shape:",summaries_f.shape,bodies_f.shape)
        t_summaries = torch.mean(summaries,dim=1)
        scores = self.score(t_summaries, bodies_f)

        print('OT topic text coverage score', scores)
        return sum(scores)/len(scores)

    def score_image(self, summaries, bodies):

        with torch.no_grad():
            # summaries_t = self.featureextractor(summaries, return_tensors = "pt")
            # summaries_f = self.clip.get_image_features(summaries_t['pixel_values'].cuda())
            
            videos = []
            for b in bodies:
                videos.append(torch.from_numpy(np.mean(np.array(b), axis=0).astype(np.float32))/ 255.0)

            bodies_t = self.featureextractor(videos, return_tensors = "pt")
            bodies_f = self.clip.get_image_features(bodies_t['pixel_values'].cuda())
        # print("video summaries shape:",summaries_f.shape,bodies_f.shape)
        # print("summaries",summaries)
        # print("bodies_f",bodies_f)
        scores = self.score(summaries, bodies_f)
        print('OT topic visual coverage score', scores)
        return sum(scores)/len(scores)

    def score_image_text(self, v_summaries, t_summaries):

        # with torch.no_grad():
        #     summaries_vt = self.featureextractor(v_summaries, return_tensors = "pt")
        #     summaries_vf = self.clip.get_image_features(summaries_vt['pixel_values'].cuda())

        #     summaries_tt = self.tokenizer(t_summaries, return_tensors = "pt", padding=True, truncation=True)
        #     summaries_tf = self.clip.get_text_features(summaries_tt['input_ids'].cuda())
        t_summaries = torch.mean(t_summaries,dim=1)
        scores = self.score(v_summaries, t_summaries)
        print('OT topic textual_visual coverage score', scores)
        return sum(scores)/len(scores)





def contrastive_loss(text_features, image_features):
    # print("contrastive loss text_features",text_features.shape) [2,30,512]
    print("contrastive loss text_features",text_features)
    print("Standard deviation:", torch.std(text_features, dim=1))
    text_features = torch.mean(text_features,dim=1)
    image_features = torch.mean(image_features,dim=1)
    print("contrastive loss text_features",text_features)
    margin=0.5
    # 计算文本特征和图像特征之间的相似度
    similarities = F.cosine_similarity(text_features.unsqueeze(1), image_features.unsqueeze(0), dim=2)
    
    # 对角线元素是正样本对的相似度
    positive_similarities = similarities.diag()

    # 计算每个正样本对与所有负样本对的损失
    loss = 0
    for i in range(len(text_features)):
        # 获取除了当前正样本对之外的所有负样本对相似度
        negative_similarities = torch.cat([similarities[i, :i], similarities[i, i+1:]])
        
        # 对于每一个负样本对，计算损失
        loss += F.relu(margin - positive_similarities[i] + negative_similarities).mean()

    return loss / len(text_features)







def save_log(log_input):
    file_name = MODEL_PATH + '/log.txt'
    p = log_input
    c = """text_file = open(file_name, "a+");text_file.write(p);text_file.close()""" 
    exec(c)


def main(args):
    ts = time.strftime('%Y-%b-%d-%H-%M-%S', time.gmtime())

    # splits = ['train', 'validation'] + (['test'] if args.test else [])
    # splits = ['train', 'test']
    splits = ['test']
    MODEL_PATH = args.save_model_path
    #wv = api.load('word2vec-google-news-300')
    dataset_folder = args.dataset_folder
    

    params = dict(
        max_summary_word=args.max_summary_word,
        max_summary_pic=args.max_summary_pic,
        text_hidden_size=args.text_hidden_size,
        video_hidden_size=args.video_hidden_size,
        #num_attention_head=args.num_attention_head,
        #num_layers=args.num_layers,
    )


    #for i in list(range(9)):
    #    train_dataset = EXMSMODataset('train_'+str(i), dataset_folder)
    #    with open('train_'+str(i)+'.pickle', 'wb') as handle:
    #        pkl.dump(train_dataset, handle , protocol=4)

    #    print("train_dataset"+str(i))

    #max_bytes = 4096
    max_bytes = 2**31 - 1
    #train_dataset_list = []

    #for i in list(range(9)):
    #    bytes_in = bytearray(0)
    #    input_size = os.path.getsize('train_'+str(i)+'.pickle')
    #    with open('train_'+str(i)+'.pickle', 'rb') as f_in:
    #        for _ in range(0, input_size, max_bytes):
    #            bytes_in += f_in.read(max_bytes)
    #    train_dataset = pkl.loads(bytes_in)
    #    train_dataset_list.extend(train_dataset)

    #val_dataset = EXMSMODataset('val', dataset_folder)
    #with open('val.pickle', 'wb') as handle:
    #    pkl.dump(val_dataset, handle , protocol=4)

    #bytes_in = bytearray(0)
    #input_size = os.path.getsize('test.pickle')
    #print("input_size", input_size)
    #with open('test.pickle', 'rb') as f_in:
     #   unpickler = pkl.Unpickler(f_in)
        # if file is not empty scores will be equal
        # to the value unpickled
    #    train_dataset = unpickler.load()

        #print("total", input_size/max_bytes)

        #for i in range(0, input_size, max_bytes):
        #    print("load data", i/max_bytes)
        #    bytes_in += f_in.read(max_bytes)
        #train_dataset =    pkl.load(f_in , protocol=4)
    #print("all loaded")
    #train_dataset = pkl.loads(bytes_in)

    def collate_func(inps):
        return [a for a in inps]

    #del bytes_in

    #val_dataset = EXMSMODataset('val_test', dataset_folder)
    train_dataset = EXMSMODataset('train', dataset_folder)
    val_dataset = EXMSMODataset('val', dataset_folder)
    test_dataset = EXMSMODataset('test', dataset_folder)
    #print("train_dataset", train_dataset)
    #print("val_dataset", val_dataset)
    print("Train Dataset size:", len(train_dataset))
    #print("Val Dataset size:", len(val_dataset))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), drop_last=True, collate_fn=collate_func)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_func)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_func)


    model = CLIPSum(**params)

    #plain_model = MMHHGATGPO(**params)
    #plain_model.load_state_dict(torch.load("/share/home/ptan6545/multimodal_OTVAE/mmhhgatgpo_model/2022-Apr-03-01:24:05/E5-1.828360.ckpt"))

    if torch.cuda.is_available():
        model = model.cuda()

    #for target_param, param in zip(model.parameters(), plain_model.parameters()):
    #    target_param.data.copy_(param.data)

    if args.resume_training:

        print(os.path.join(MODEL_PATH,args.model_name))
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH,args.model_name)))
        model.train()


    print(model)
    
    # 只冻结CLIP模型部分
    for param in model.clip.parameters():
        param.requires_grad = False
    for param in model.cliptext.parameters():
        param.requires_grad = False

    save_model_path = os.path.join(args.save_model_path, ts)
    # save_model_path = os.path.join(args.save_model_path, '1')
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def loss_fn(output_text_summaries_word, output_text_summaries, output_video_summaries, texts, videos, OT_topic):
        
        
        # cut-off unnecessary padding from target, and flatten
        #logp = logp[:, :torch.max(summary_length).item(), :].view(-1, logp.size(2))
        #logp = logp[:, :summary_length, :].contiguous().view(-1, logp.size(2))
        #text_logp = text_logp[:, :, :].contiguous().view(-1, text_logp.size(2))
        #print("loss target", target.size())
        #print("loss logp", logp.size())
        # Negative Log Likelihood

        #textcoverage_loss = textCoverageLoss.score(output_text_summaries, texts)
        textcoverage_loss = OT_topic.score_text(output_text_summaries, texts)
        #videocoverage_loss = VideoCoverageLoss(output_video_summaries, videos)
        videocoverage_loss = OT_topic.score_image(output_video_summaries, videos)
        #NLL_loss = NLL(logp, target)
        #print("text_z", text_z.size())
        #print("video_z", video_z.size())
        #print('target', Variable(torch.ones(text_z.size()[0])).size())
        #mmcoverage_loss = mmCoverageLoss.score(output_text_summaries, output_video_summaries, texts, videos)
        #mmcoverage_loss = mmCoverageLoss.score(output_text_summaries, output_video_summaries)
        mmcoverage_loss = OT_topic.score_image_text(output_video_summaries, output_text_summaries)
        fluency_loss, _ = fluencyLoss.score(output_text_summaries_word, output_video_summaries, descriptions, videos)
        # KL Divergence


        return textcoverage_loss, videocoverage_loss , mmcoverage_loss, sum(fluency_loss)/len(fluency_loss)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    fluency_news_model_file = os.path.join("models", "gpt2_copier23.bin")

    #textCoverageLoss = TextCoverageLoss()
    #mmCoverageLoss = MmCoverageLoss()
    #mmCoverageLoss = MmAlignmentLoss()
    # TODO
    mutual_info_loss =  None
    fluencyLoss = GeneTransformer(max_output_length=args.max_summary_word, device="cuda", starter_model=fluency_news_model_file)
    otTopic = OT_topic()
    writer = SummaryWriter()
    

    test_criterion = nn.MSELoss()
    initial_state = {name: param.clone().detach() for name, param in model.named_parameters()}
    for epoch in range(args.epochs):
        
        for split in splits:
            print("split", split)
            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()


                data_loader = train_data_loader
            else:
                model.eval()
                data_loader = test_data_loader
            
            #print("data_loader", len(data_loader))
            
            for iteration, data in enumerate(data_loader):
                gc.collect()
                torch.cuda.empty_cache()
                #print("iteration", iteration)
                #print("data", data)
                print("epoch ", epoch)
                print("iteration ", iteration)
                file_id = []
                descriptions = []
                videos = []
                #titles = []
                #scenes = []


                for d in data:
                    file_id.append(d[0])
                    descriptions.append(d[1])
                    videos.append(d[2])
                    #titles.append(d[3])
                    #scenes.append(d[6])


                batch_size = len(data)

                # Forward pass
                #bodies = [doc[args.dataset_doc_field] for doc in documents]
                output_text_summaries, output_text_summaries_pos, text_logp, output_video_summaries, output_video_summaries_pos, video_logp, video_mutual_info, text_mutual_info ,summary_image_feature, summary_sent_feature =  model(descriptions, videos)
                # print("length of output video tensor", output_video_summaries[0].shape)
                # print("output video tensor", output_video_summaries)
                # Write results to the file

                
                for idx, id in enumerate(file_id):
                    
                    dir1=split
            
                    dir = os.path.join('output',dir1)
                    dir = os.path.join(dir,id+"_"+str(epoch))
                    dir_image=dir+'.png'
                    dir_text=dir+'.txt'
                    print(dir_image+"  "+dir_text)
                    

                    image = output_video_summaries[idx].cpu().numpy()
                    image = image[...,::-1]
        
                    image_pil = Image.fromarray(image)
                    image_pil.save(dir_image)

                    with open(dir_text,"w") as f:
                        f.write(output_text_summaries[idx])

                # 计算损失
                # contra_loss = contrastive_loss(video_mutual_info, text_mutual_info)

                # loss calculation
                textcoverage_loss, videocoverage_loss, mmalignment_loss, fluency_loss = loss_fn(output_text_summaries, summary_sent_feature, summary_image_feature, descriptions, videos, otTopic)

                #loss = torch.zeros(1, requires_grad=True)

                #loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                print("textcoverage_loss", textcoverage_loss)
                print("videocoverage_loss", videocoverage_loss)
                print("mmalignment_loss", mmalignment_loss)
                print("fluency_loss", fluency_loss)
                # print("contra_loss",contra_loss)
                #print("batch_size", batch_size)
                #loss = (textcoverage_loss + videocoverage_loss + mmalignment_loss+text_KL_weight * text_KL_loss+video_KL_loss*video_KL_weight) / batch_size
                loss = textcoverage_loss + 0.01 * videocoverage_loss + mmalignment_loss + 2* fluency_loss#previous videocoverage 0.001
                # print("test_feature",test_feature.shape)
                # loss = test_criterion(video_feature, test_feature)
                # print(loss)
                # loss = videocoverage_loss
                
 
                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    print(loss)
                    # print("summary_image_feature grad",model.image_pret_transformer_model.encoder.layers[0].linear2.weight.grad)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    step += 1
                    # 检查权重是否更新
                    # for name, param in model.named_parameters():
                    #     initial_param = initial_state[name]
                    #     # 检查是否有任何变化
                    #     if not torch.equal(initial_param, param):
                    #         print(f"Parameter {name} has changed.")
                    #     else:
                    #         print(f"Parameter {name} has not changed.")
                    # for name, param in model.named_parameters():
                    #     if param.requires_grad:
                    #         print(f"{name} - mean: {param.data.mean()}, std: {param.data.std()}")


                # bookkeepeing
                tracker['LOSS'] = torch.cat((tracker['LOSS'], loss.data.view(1, -1)), dim=0)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/Loss" % split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Text Coverage Loss" % split.upper(), textcoverage_loss,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Video Coverage Loss" % split.upper(), videocoverage_loss,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/MM Coverage Loss" % split.upper(), mmalignment_loss,
                                      epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, Text-Coverage-Loss %9.4f, Video-Coverage-Loss %9.4f, MM-Coverage-Loss %9.4f"
                          % (split.upper(), iteration, len(data_loader)-1, loss.item(), textcoverage_loss, videocoverage_loss, mmalignment_loss))

                if split == 'valid':
                    print("validing...")
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    #tracker['target_sents'] += idx2word(answer, i2w=datasets['train'].get_i2w(),pad_idx=PAD)
                    tracker['target_sents'] += idx2word(answer, i2w=self.id2word,pad_idx=PAD)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)
                    
                if iteration%200==0 and split =='train':
                    checkpoint_path = os.path.join(save_model_path, "noise_filter_E%i-%9f.ckpt" % (iteration,tracker['LOSS'].mean()))
                    torch.save(model.state_dict(), checkpoint_path)
                    print("Model saved at %s" % checkpoint_path)

            print("%s Epoch %02d/%i, Mean LOSS %9.4f" % (split.upper(), epoch, args.epochs, tracker['LOSS'].mean()))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/LOSS" % split.upper(), torch.mean(tracker['LOSS']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json' % epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train' and epoch%5 ==0:
                checkpoint_path = os.path.join(save_model_path, "E%i-%9f.ckpt" % (epoch,tracker['LOSS'].mean()))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)



# def sparse_ot(weights1, weights2, M):
#     """ Compute Wasserstein distances"""
    
#     weights1 = weights1/weights1.sum()
#     weights2 = weights2/weights2.sum()
    
#     active1 = np.where(weights1)[0]
#     active2 = np.where(weights2)[0]
    
#     weights_1_active = weights1[active1]
#     weights_2_active = weights2[active2]
#     #print("active1", active1)
#     #print("active2", active2)
#     #print("M", M)
#     #print("M", M)
#     try1 = M[active1][:,active2]
#     #print("try1", try1)
#     M_reduced = np.ascontiguousarray(M[active1][:,active2])
    
#     return ot.emd2(weights_1_active,weights_2_active,M_reduced)


def sparse_ot(weights1, weights2, M):
    """Compute Wasserstein distances for one-dimensional weight vectors using PyTorch."""
    # 确保所有张量都在相同的设备上
    device = weights1.device
    weights2 = weights2.to(device)
    M = M.to(device)
    
    weights1 = F.softplus(weights1)
    weights2 = F.softplus(weights2)
    # 归一化权重
    weights1 = (weights1 + 1e-8) / weights1.sum()
    weights2 = (weights2 + 1e-8) / weights2.sum()


    # 找到非零元素的索引
    active1 = weights1.nonzero(as_tuple=False).squeeze()
    active2 = weights2.nonzero(as_tuple=False).squeeze()

    # 选择激活的权重和成本矩阵
    weights_1_active = weights1[active1]
    weights_2_active = weights2[active2]
    M_reduced = M[active1][:, active2]

    # 计算 Wasserstein 距离
    dist = sinkhorn_distance(weights_1_active, weights_2_active, M_reduced)

    return dist


def sinkhorn_distance(weights1, weights2, M, epsilon=0.01, max_iter=100):
    """简化的 Sinkhorn 算法实现."""
    K = torch.exp(-M / epsilon)
    u = torch.ones_like(weights1) / weights1.size(0)
    for _ in range(max_iter):
        u = weights1 / torch.mv(K, weights2 / torch.mv(K.t(), u))
    v = weights2 / torch.mv(K.t(), u)
    return torch.sum(u * torch.mv(K * M, v))



def construct_BOW(tokens):
    bag_vector = np.zeros(BERT_NUM_TOKEN)        
    for token in tokens:            
        bag_vector[token] += 1                            
    return bag_vector/len(tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--max_sequence_length', type=int, default=900)
    parser.add_argument('--max_article_length', type=int, default=5)
    parser.add_argument('--max_summary_pic', type=int, default=1)
    parser.add_argument('--max_summary_word', type=int, default=12)

    parser.add_argument('--test', action='store_true', default='False')

    parser.add_argument('-ep', '--epochs', type=int, default=100000)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-ths', '--text_hidden_size', type=int, default=128)
    parser.add_argument('-vhs', '--video_hidden_size', type=int, default=128)
    parser.add_argument('-nah', '--num_attention_head', type=int, default=2)
    parser.add_argument('-nl', '--num_layers', type=int, default=2)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='multimodal_model')
    parser.add_argument('--dataset_folder', type=str, help='folder of dataset', default='data')
    parser.add_argument("--resume_training", type=bool, default=False) 
    parser.add_argument("--model_name", type=str, default='bench') 
    args = parser.parse_args()


    main(args)



