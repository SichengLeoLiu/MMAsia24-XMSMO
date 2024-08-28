import numpy as np
import re, math
import pickle as pkl
from torch.nn import functional as F
#from transformers import BertTokenizer, BertModel
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils
import time
from utils import PAD, UNK, END, START, UNK_TOKEN, PAD_TOKEN
from torch.nn.functional import softplus
import torchvision.models as models
import nltk
from torchvision import transforms
from PIL import Image
from nltk import word_tokenize
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPTextModel
from nltk.tokenize import sent_tokenize
from transformers.pipelines import pipeline
nltk.download('punkt')

def split_feature(video_embeddings, text_embeddings, batch_size, hidden_size):
    device = video_embeddings.device 
    # 确保视频和文本嵌入的batch size是相同的
    assert video_embeddings.size(0) == text_embeddings.size(0)
    
    # 获取嵌入维度
    v_b, v_l, v_h = video_embeddings.shape
    t_b, t_l, t_h = text_embeddings.shape
    assert v_h == t_h  # 确保视频和文本嵌入在特征维度上是相同的

    # 初始化列表来存储共有信息和各自模态的独有信息

    video_mutual_info = [[] for i in range(batch_size)]
    text_mutual_info = [[] for i in range(batch_size)]
    video_only_info = [[] for i in range(batch_size)]
    text_only_info = [[] for i in range(batch_size)]

    # 对每个batch进行处理
    for batch_idx in range(v_b):
        # 计算当前batch中所有特征对的相似度
        similarities = torch.zeros((v_l, t_l), device = device)
        for vl in range(v_l):
            for tl in range(t_l):
                # 计算视频特征和文本特征之间的余弦相似度
                cos_sim = F.cosine_similarity(video_embeddings[batch_idx][vl].unsqueeze(0), text_embeddings[batch_idx][tl].unsqueeze(0))
                similarities[vl, tl] = cos_sim

        # 将相似度矩阵转换为一维数组并排序，选择相似度最高的50%
        flat_similarities = similarities.view(-1)
        sorted_similarities, sorted_indices = torch.sort(flat_similarities, descending=True)
        top_half_indices = sorted_indices[:len(sorted_indices) // 2]



        # 选取前50%的相似度对应的特征
        selected_video_indices = top_half_indices // t_l
        selected_text_indices = top_half_indices % t_l


        selected_video_indices=torch.unique(selected_video_indices, sorted=False)[:vl//2]

        selected_text_indices=torch.unique(selected_text_indices, sorted=False)[:tl//2]

        for vi in range(v_l):
            if vi in selected_video_indices:
                video_mutual_info[batch_idx].append(video_embeddings[batch_idx, vi])
            else:
                video_only_info[batch_idx].append(video_embeddings[batch_idx, vi])
        for ti in range(t_l):
            if ti in selected_text_indices:
                text_mutual_info[batch_idx].append(text_embeddings[batch_idx, ti])
            else:
                text_only_info[batch_idx].append(text_embeddings[batch_idx, ti])


        
    max_length = max(len(sublist) for sublist in video_mutual_info)
    for batch_id in range(batch_size):
        if len(video_mutual_info[batch_id])<max_length:
            for i in range(len(video_mutual_info[batch_id]),max_length):
                video_mutual_info[batch_id].append(torch.zeros([hidden_size], device = device))

    max_length = max(len(sublist) for sublist in text_mutual_info)
    for batch_id in range(batch_size):
        if len(text_mutual_info[batch_id])<max_length:
            for i in range(len(text_mutual_info[batch_id]),max_length):
                text_mutual_info[batch_id].append(torch.zeros([hidden_size], device = device))

    max_length = max(len(sublist) for sublist in text_only_info)
    for batch_id in range(batch_size):
        if len(text_only_info[batch_id])<max_length:
            for i in range(len(text_only_info[batch_id]),max_length):
                text_only_info[batch_id].append(torch.zeros([hidden_size], device = device))

        
    max_length = max(len(sublist) for sublist in video_only_info)
    for batch_id in range(batch_size):
        if len(video_only_info[batch_id])<max_length:
            for i in range(len(video_only_info[batch_id]),max_length):
                video_only_info[batch_id].append(torch.zeros([hidden_size], device = device))

    video_mutual_info = torch.stack([torch.stack(sublist) for sublist in video_mutual_info])
    text_mutual_info = torch.stack([torch.stack(sublist) for sublist in text_mutual_info])
    text_only_info = torch.stack([torch.stack(sublist) for sublist in text_only_info])
    video_only_info = torch.stack([torch.stack(sublist) for sublist in video_only_info])


    return video_mutual_info, text_mutual_info, video_only_info, text_only_info

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CLIPSum(nn.Module):
    def __init__(self, text_hidden_size, video_hidden_size, 
                max_summary_word, max_summary_pic):

        super(CLIPSum, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_summary_word = max_summary_word
        self.max_summary_pic = max_summary_pic
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.featureextractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        #self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        #self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        #self.text_model.to(device)
        #self.vision_model.eval()
        #self.vision_model.to(device)
        

        self.cliptext = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.cliptext.eval()
        self.hidden_size = 512
        self.topic_size = 100
        self.dropout = 0.1
        self.num_attention_head = 16

        self.word_pret_pos_encoder = PositionalEncoding(self.hidden_size)
        self.word_pret_transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)

        self.word_pos_encoder = PositionalEncoding(self.hidden_size)
        self.word_transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)

        self.image_pret_pos_encoder = PositionalEncoding(self.hidden_size)
        self.image_pret_transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)

        self.image_pos_encoder = PositionalEncoding(self.hidden_size)
        self.image_transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)


        self.v2tattn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_head, batch_first=True, dropout=self.dropout)
        self.v2tattn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.v2tattn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v2tattn_linear_layer_norm = nn.LayerNorm(self.hidden_size)
        
        self.t2vattn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_head, batch_first=True, dropout=self.dropout)
        self.t2vattn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.t2vattn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.t2vattn_linear_layer_norm = nn.LayerNorm(self.hidden_size)

        self.coattn = nn.Linear(self.hidden_size*2, self.hidden_size)

        with open("kmeans_model_100.pkl", "rb") as f:
            self.k_means =  pkl.load(f)

        self.outputs2vocab = nn.Linear(self.hidden_size , 1)


        self.outputs2coverframe = nn.Linear(self.hidden_size, 1)

        self.fliter_linear1= nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fliter_linear2= nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fliter_linear3= nn.Linear(self.hidden_size, self.hidden_size)
        self.filter_sigmoid = nn.Sigmoid()
        

    def forward(self, input_text, input_video, text_summary_length=11):

        batch_size = len(input_text)
        
        batch_sent = []
        batch_sent_num = []
        batch_sent_pad = []
        text_token_list = []
        text_id_list = []
        for text in input_text:
            text = text.replace("Please subscribe HERE http://bit.ly/1rbfUog", "")
            text = text.replace("#BBCNews", "")
            text = text.replace("\n", " \n ")
            sents = sent_tokenize(text)
            text_id_l = []
            text_token_l = []
            for sent in sents:
                text_t = word_tokenize(sent)
                for t in text_t:
                    text_token = self.tokenizer(t , return_tensors = "pt", padding=True, truncation=True, max_length=50)
                    text_id_l.append(int(text_token['input_ids'][0][1]))
                text_token_l.extend(text_t)
            text_token_list.append(text_token_l)
            print("text_id_l", text_id_l)
            text_id_list.append(torch.LongTensor(text_id_l[:77])) #clip max length

            batch_sent.append(sents)
            batch_sent_num.append(len(sents))

        for batch in batch_sent:
            batch += [' '] * (max(batch_sent_num) - len(batch))
            batch_sent_pad.append(batch)


        # Get Text features from CLIP
        batch_sent_pad_t = [list(x) for x in zip(*batch_sent_pad)]
        sent_feature = []
        word_feature = []
        pad_sent_len = []
        sent_feature = []
        last_len = 0
        #for sent in batch_sent_pad_t:
        #print("sent", sent)
        text_id_list = torch.nn.utils.rnn.pad_sequence(text_id_list, batch_first=True).cuda()
        # print("text_id_list", text_id_list.size()) # [2,77]
        word_feature = self.cliptext(text_id_list).last_hidden_state
        word_feature = self.word_pret_pos_encoder(word_feature)
        #word_feature = self.word_pret_transformer_model(word_feature.permute(1,0,2)).permute(1,0,2)
        text_b,text_s,text_d = word_feature.size()

        # print("word_feature", word_feature.size()) # [2,77,512]



        # Get video features from CLIP
        if batch_size > 1:
            input_video = torch.nn.utils.rnn.pad_sequence(input_video, batch_first=True)

        scene_frame_pad_batch_list = input_video[:, :60,:,:,:] #max for cuda
        
        v_image_features = []

        for i, images in enumerate(scene_frame_pad_batch_list.permute(1,0,4,2,3)):

            image_batch = []
            for batch in images:
                image_batch.append(transforms.ToPILImage()(batch.squeeze_(0)))
            image_token = self.featureextractor(image_batch, return_tensors = "pt")
            v_image_features.append(self.clip.get_image_features(image_token['pixel_values'].cuda()))


        image_feature = torch.stack(v_image_features, dim=1)
        image_feature = self.image_pret_pos_encoder(image_feature)
        # print("image_feature", image_feature.size()) # [2,60,512]

        video_mutual_info, text_mutual_info, video_only_info, text_only_info=split_feature(image_feature,word_feature,text_b,text_d)
        print("filter 2")
        # TODO
        # multimodal_mutual_info = torch.cat((video_mutual_info, text_mutual_info),dim=1)
        # multimodal_mutual_info=torch.mean(multimodal_mutual_info,dim=1)
        # mutual_info_expanded = multimodal_mutual_info.unsqueeze(1).expand(-1, video_only_info.size(1), -1)
        
        # video_only_info_gate=self.fliter_linear1(torch.cat((video_only_info,mutual_info_expanded),dim=2))
        # video_only_info_gate=self.filter_sigmoid(video_only_info_gate)
        # video_only_info=video_only_info*video_only_info_gate

        # mutual_info_expanded = multimodal_mutual_info.unsqueeze(1).expand(-1, text_only_info.size(1), -1)

        # text_only_info_gate=self.fliter_linear2(torch.cat((text_only_info,mutual_info_expanded),dim=2))
        # text_only_info_gate=self.filter_sigmoid(text_only_info_gate)
        # text_only_info=text_only_info*text_only_info_gate

        # word_feature = torch.cat((text_mutual_info,text_only_info),dim=1)

        # image_feature = torch.cat((video_mutual_info,video_only_info),dim=1)
        
        multimodal_mutual_info = torch.cat((video_mutual_info, text_mutual_info),dim=1)
        multimodal_mutual_info=torch.mean(multimodal_mutual_info,dim=1)
        mutual_info_expanded = multimodal_mutual_info.unsqueeze(1).expand(-1, image_feature.size(1), -1)
        
        video_only_info_gate=self.fliter_linear1(torch.cat((image_feature,mutual_info_expanded),dim=2))
        video_only_info_gate=self.filter_sigmoid(video_only_info_gate)
        image_feature=image_feature*video_only_info_gate*2

        mutual_info_expanded = multimodal_mutual_info.unsqueeze(1).expand(-1, word_feature.size(1), -1)

        text_only_info_gate=self.fliter_linear2(torch.cat((word_feature,mutual_info_expanded),dim=2))
        text_only_info_gate=self.filter_sigmoid(text_only_info_gate)
        word_feature=word_feature*text_only_info_gate*2
        

        # word_feature = torch.cat((text_mutual_info,text_only_info),dim=1)

        # image_feature = torch.cat((video_mutual_info,video_only_info),dim=1)





        # Get sentence features
        sent_feature = self.clip.get_text_features(text_id_list)
        # print("sent_feature", sent_feature.size()) # [2,512]
        
        # Get topics from sentences
        topic_distance_t = torch.from_numpy(self.k_means.transform(sent_feature.detach().cpu().numpy())).cuda()
        topic_distance_t_target = torch.zeros(text_b, 512).cuda()
        topic_distance_t_target[:, :self.topic_size] = topic_distance_t
        word_feature=torch.cat([topic_distance_t_target.unsqueeze(1), word_feature], dim=1)
        # word_feature = self.word_pret_transformer_model(word_feature, topic_distance_t_target.unsqueeze(1).expand(-1,text_s , -1))
        word_feature = self.word_pret_transformer_model.encoder(word_feature)
        word_feature=word_feature[:,1:,:]
        
        # Get video features
        video_feature = torch.mean(image_feature, dim=1)
        print("video_feature", video_feature.size()) # [2,512]

        # Get topics from videos
        video_b,video_s,video_d = image_feature.size()
        topic_distance_v = torch.from_numpy(self.k_means.transform(video_feature.detach().cpu().numpy())).cuda()
        print("topic_distance_v",topic_distance_v.shape)

        topic_distance_v_target = torch.zeros(video_b, 512).cuda()
        topic_distance_v_target[:, :self.topic_size] = topic_distance_v

        image_feature = torch.cat([topic_distance_v_target.unsqueeze(1), image_feature], dim=1)
        # image_feature = self.image_pret_transformer_model(image_feature, topic_distance_v_target.unsqueeze(1).expand(-1,video_s , -1))
        image_feature = self.image_pret_transformer_model.encoder(image_feature)
        image_feature=image_feature[:,1:,:]


        v2t_attn, _ = self.v2tattn(image_feature, word_feature, word_feature)
        v2t_attn = self.v2tattn_layer_norm(v2t_attn) + image_feature
        v2t_attn_linear = self.v2tattn_linear(v2t_attn.reshape(-1, video_d))
        v2t_attn = self.v2tattn_linear_layer_norm(v2t_attn_linear.view(video_b,video_s,video_d)) + v2t_attn

        t2v_attn, _ = self.t2vattn(word_feature, image_feature, image_feature)
        t2v_attn = self.t2vattn_layer_norm(t2v_attn) + word_feature
        t2v_attn_linear = self.t2vattn_linear(t2v_attn.reshape(-1, text_d))
        t2v_attn = self.t2vattn_linear_layer_norm(t2v_attn_linear.view(text_b,text_s,text_d)) + t2v_attn

        # print("v2t_attn", v2t_attn.size()) # [2,60,512]
        # print("t2v_attn", t2v_attn.size()) # [2,77,512]
        #print("torch.cat([v2t_attn.squeeze(1), t2v_attn.squeeze(1)],dim=1)", torch.cat([v2t_attn.squeeze(1), t2v_attn.squeeze(1)],dim=1).size())
        overall_feature = self.coattn(torch.cat([torch.mean(v2t_attn, dim=1), torch.mean(t2v_attn, dim=1)],dim=1)) 
        # print("overall_feature",overall_feature.shape) # [2,512]


        #overall_feature = torch.mean(torch.stack([video_feature, sent_feature],dim=1), dim=1)

        
        # topic_distance = torch.from_numpy(self.k_means.transform(overall_feature.detach().cpu().numpy())).cuda()

        # topic_distance_target = torch.zeros(text_b, 512).cuda()
        # topic_distance_target[:, :self.topic_size] = topic_distance
        # print("topic",topic_distance.shape) # [2,100]
        # print("topic",topic_distance_target.unsqueeze(1).expand(-1,text_s , -1).shape) # [2,60,512]
        # word_feature = self.word_pos_encoder(word_feature)
        # word_attn_output = self.word_transformer_model(word_feature, topic_distance_target.unsqueeze(1).expand(-1,text_s , -1))
        word_feature=torch.cat([overall_feature.unsqueeze(1), word_feature], dim=1)
        word_attn_output = self.word_transformer_model.encoder(word_feature)
        # word_attn_output = self.word_transformer_model(word_feature, overall_feature.unsqueeze(1).expand(-1,text_s , -1))
        # print("word_attn_output", word_attn_output.size()) # [2,77,512]

        text_logp = self.outputs2vocab(word_attn_output.reshape(-1, word_attn_output.size(2)))

        text_logp = text_logp.view(text_b, text_s+1, 1)
        text_logp=text_logp[:,1:,:]
        # image_feature = self.image_pos_encoder(image_feature)

        image_feature = torch.cat([overall_feature.unsqueeze(1), image_feature], dim=1)
        # image_attn_output = self.image_transformer_model(image_feature, topic_distance_target.unsqueeze(1).expand(-1,video_s , -1))
        image_attn_output = self.image_transformer_model.encoder(image_feature)
        # print(image_feature)
        # image_attn_output = self.image_transformer_model(image_feature, overall_feature.unsqueeze(1).expand(-1,video_s , -1))
        # print(image_attn_output)
        # print("image_attn_output", image_feature.size()) # [2,60,512]

        video_logp = self.outputs2coverframe(image_attn_output.reshape(-1, image_attn_output.size(2)))
        
        # print("video_logp: ",video_logp.size()) 
        video_logp = video_logp.view(video_b, video_s+1, 1)
        video_logp = video_logp[:,1:,:]
        # print("video_logp: ",video_logp.size()) # [2,60,1]


        output_video_summaries = []
        output_video_summaries_pos = []

        for image, summary in zip(scene_frame_pad_batch_list, video_logp):
            rank = torch.argsort(summary, dim=0, descending=True)
            print("image", len(image))
            output_video_summaries_pos.append(rank[0])
            output_video_summaries.append(image[int(rank[0])])
        

        output_text_summaries = []
        output_text_summaries_pos = []

        for text, t_id, summary in zip(text_token_list, text_id_list, text_logp):
            word_count = 0
            pos = []
            rank = torch.argsort(summary, dim=0, descending=True)
            #print("rank text", rank)
            text_id = []

            filtered_rank = []
            #for i in sorted(rank):
            for i in rank:
                if i < len(text) and t_id[int(i)] < 49406:
                    filtered_rank.append(int(i))
                    if text[int(i)] != PAD_TOKEN and text[int(i)].isalnum():
                        word_count += 1
                
                if word_count > self.max_summary_word:
                    break

            for i in sorted(filtered_rank):
                text_id.append(text[int(i)])
                pos.append(int(i))

            #print("text_id", text_id)
            output_text_summaries.append(" ".join(text_id))
            output_text_summaries_pos.append(pos)
                    
        print("output_text_summaries", output_text_summaries)
        print("output_text_summaries_pos", output_text_summaries_pos)
        print("output_video_summaries_pos",output_video_summaries_pos)
        
        
        return output_text_summaries, output_text_summaries_pos, text_logp, output_video_summaries, output_video_summaries_pos, video_logp
