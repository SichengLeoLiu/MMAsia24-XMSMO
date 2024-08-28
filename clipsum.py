# import numpy as np
# import re, math
# import pickle as pkl
# from torch.nn import functional as F
# #from transformers import BertTokenizer, BertModel
# import os
# import torch.nn.functional as F
# import torch.nn as nn
# import torch
# import torch.nn.utils.rnn as rnn_utils
# import time
# from utils import PAD, UNK, END, START, UNK_TOKEN, PAD_TOKEN
# from torch.nn.functional import softplus
# import torchvision.models as models
# import nltk
# from torchvision import transforms
# from PIL import Image
# from nltk import word_tokenize
# from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPTextModel
# from nltk.tokenize import sent_tokenize
# from transformers.pipelines import pipeline
# from nltk.tag import pos_tag
# import cv2
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

# class CLIPSum(nn.Module):
#     def __init__(self, text_hidden_size, video_hidden_size, 
#                 max_summary_word, max_summary_pic):

#         super(CLIPSum, self).__init__()
#         self.pos = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']
        
#         self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.max_summary_word = max_summary_word
#         self.max_summary_pic = max_summary_pic
        
#         self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#         self.featureextractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
#         #self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
#         #self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         #self.text_model.to(device)
#         #self.vision_model.eval()
#         #self.vision_model.to(device)
        

#         self.cliptext = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
#         #self.cliptext.eval()
#         #self.clipvision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.clip.eval()
#         self.cliptext.eval()
#         #self.clipvision.eval()
#         self.hidden_size = 512
#         self.topic_size = 100
#         self.dropout = 0.1
#         #self.image_hidden_size=768
#         self.num_attention_head = 16
#         #self.wordattn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_head, dropout=self.dropout)
#         #self.word_layer_norm = nn.LayerNorm(self.hidden_size)
#         #self.wordlstm= nn.LSTM(self.hidden_size*2, self.hidden_size, num_layers=2, bidirectional=False, batch_first = True)

#         self.word_pret_pos_encoder = PositionalEncoding(self.hidden_size)
#         #word_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16)
#         #self.word_pret_transformer_model = nn.TransformerEncoder(word_encoder_layer, num_layers=12)
#         self.word_pret_transformer_model = nn.Transformer(self.hidden_size, nhead=self.num_attention_head, num_encoder_layers=12, batch_first=True)

#         self.word_modalspecific_transformer_model = nn.Transformer(self.hidden_size, nhead=self.num_attention_head, num_encoder_layers=12, batch_first=True)

#         #self.word_pos_encoder = PositionalEncoding(self.hidden_size + 1)
#         self.word_transformer_model = nn.Transformer(self.hidden_size, nhead=self.num_attention_head, num_encoder_layers=12, batch_first=True)

#         self.image_pret_pos_encoder = PositionalEncoding(self.hidden_size)
#         #image_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16)
#         #self.image_pret_transformer_model = nn.TransformerEncoder(image_encoder_layer, num_layers=12)
#         self.image_pret_transformer_model = nn.Transformer(self.hidden_size, nhead=self.num_attention_head, num_encoder_layers=12, batch_first=True)
#         self.image_modalspecific_transformer_model = nn.Transformer(self.hidden_size, nhead=self.num_attention_head, num_encoder_layers=12, batch_first=True)

#         #self.image_pos_encoder = PositionalEncoding(self.hidden_size + 1)
#         self.image_transformer_model = nn.Transformer(self.hidden_size, nhead=self.num_attention_head, num_encoder_layers=12, batch_first=True)


#         self.v2tattn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_head, batch_first=True, dropout=self.dropout)
#         self.v2tattn_layer_norm = nn.LayerNorm(self.hidden_size)
#         self.v2tattn_linear = nn.Linear(self.hidden_size, self.hidden_size)
#         self.v2tattn_linear_layer_norm = nn.LayerNorm(self.hidden_size )
        
#         self.t2vattn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_head, batch_first=True, dropout=self.dropout)
#         self.t2vattn_layer_norm = nn.LayerNorm(self.hidden_size)
#         self.t2vattnn_linear = nn.Linear(self.hidden_size , self.hidden_size )
#         self.t2vattn_linear_layer_norm = nn.LayerNorm(self.hidden_size )

#         self.coattn = nn.Linear((self.hidden_size )*2, self.hidden_size)
#         #self.image_layer_norm = nn.LayerNorm(self.hidden_size)
#         #self.imagelstm= nn.LSTM(self.hidden_size*2, self.hidden_size, num_layers=2, bidirectional=False, batch_first = True)
        

#         with open("kmeans_model_100.pkl", "rb") as f:
#             self.k_means =  pkl.load(f)

#         self.outputs2vocab = nn.Linear(self.hidden_size, 1)

#        # self.pic_linear_z = nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size)

#         self.outputs2coverframe = nn.Linear(self.hidden_size, 1)
        

#     def forward(self, input_text, input_video, text_summary_length=11):

#         batch_size = len(input_text)
        
#         batch_sent = []
#         batch_sent_num = []
#         batch_sent_pad = []
#         text_token_list = []
#         text_id_list = []
#         pos_list = []
#         # preprocess for text
#         for text in input_text:
#             text = text.replace("Please subscribe HERE http://bit.ly/1rbfUog", "")
#             text = text.replace("#BBCNews", "")
#             text = text.replace("\n", " \n ")
#             sents = sent_tokenize(text)
#             text_id_l = []
#             text_token_l = []
#             pos_l = []
#             for sent in sents:
#                 text_t = word_tokenize(sent)
#                 pos_parse = nltk.pos_tag(text_t)
#                 pos_t = []
#                 for p in pos_parse:
#                     # print("p", p)
#                     try:
#                         pos_t.append(self.pos.index(p[1])+1)
#                     except:
#                         pos_t.append(0)
#                 for t in text_t:
#                     text_token = self.tokenizer(t , return_tensors = "pt", padding=True, truncation=True, max_length=50)
#                     text_id_l.append(int(text_token['input_ids'][0][1]))
#                 text_token_l.extend(text_t)
#                 pos_l.extend(pos_t)
#             text_token_list.append(text_token_l)
#             # print("text_id_l", text_id_l)
#             pos_list.append(torch.LongTensor(pos_l[:77]))
#             text_id_list.append(torch.LongTensor(text_id_l[:77])) #clip max length

#             batch_sent.append(sents)
#             batch_sent_num.append(len(sents))

#         for batch in batch_sent:
#             batch += [' '] * (max(batch_sent_num) - len(batch))
#             batch_sent_pad.append(batch)

#         batch_sent_pad_t = [list(x) for x in zip(*batch_sent_pad)]
#         sent_feature = []
#         word_feature = []
#         pad_sent_len = []
#         sent_feature = []
#         last_len = 0
#         #for sent in batch_sent_pad_t:
#         #print("sent", sent)
#         pos_id_list = torch.nn.utils.rnn.pad_sequence(pos_list, batch_first=True).cuda()
#         text_id_list = torch.nn.utils.rnn.pad_sequence(text_id_list, batch_first=True).cuda()
#         # print("text_id_list", text_id_list.size())
#         # print("pos_id_list", pos_id_list.size())
#         word_feature = self.cliptext(text_id_list).last_hidden_state
#         word_feature = self.word_pret_pos_encoder(word_feature)
#         #word_feature = self.word_pret_transformer_model(word_feature.permute(1,0,2)).permute(1,0,2)
        

#         print("word_feature", word_feature.size())
#         #text_overall_feature = self.clip.get_text_features(text_token['input_ids'].cuda())
#         #text_token_list.append(text_token_l)
#         sent_feature = self.clip.get_text_features(text_id_list)
#         print("sent_feature", sent_feature.size())

#         text_b,text_s,text_d = word_feature.size()

#         pos_id_target = torch.zeros(text_b, text_s, self.hidden_size).cuda()
#         pos_id_target[:, :, :1] = pos_id_list.unsqueeze(2)

#         #word_feature = torch.cat([word_feature, pos_id_list], dim=2)

#         word_feature = self.word_modalspecific_transformer_model(word_feature, pos_id_target)



        
#         topic_distance_t = torch.from_numpy(self.k_means.transform(sent_feature.detach().cpu().numpy())).cuda()

#         topic_distance_t_target = torch.zeros(text_b, self.hidden_size).cuda()
#         topic_distance_t_target[:, :self.topic_size] = topic_distance_t
        
#         word_feature = self.word_pret_transformer_model(word_feature, topic_distance_t_target.unsqueeze(1).expand(-1,text_s , -1))

#         last_len = 0

#         if batch_size > 1:
            
#             input_video = torch.nn.utils.rnn.pad_sequence(input_video, batch_first=True)

#         scene_frame_pad_batch_list = input_video[:, :60,:,:,:] #max for cuda

#         v_image_features = []
#         v_num_face = []
#         for i, images in enumerate(scene_frame_pad_batch_list.permute(1,0,4,2,3)):
#             num_faces = []
#             image_batch = []
#             for batch in images:
#                 img = transforms.ToPILImage()(batch.squeeze_(0))
#                 #print("img", img.size())
#                 image_batch.append(img)
#                 p#rint("img", img.size())
#                 faces = self.face_cascade.detectMultiScale(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY), 1.1, 4)
#                 num_faces.append(len(faces))
#             v_num_face.append(torch.LongTensor(num_faces))
#             image_token = self.featureextractor(image_batch, return_tensors = "pt")
#             v_image_features.append(self.clip.get_image_features(image_token['pixel_values'].cuda()))

#         v_num_face = torch.stack(v_num_face, dim=1)
#         # print("v_num_face", v_num_face.size())
#         # print("v_num_face", v_num_face)
#         image_feature = torch.stack(v_image_features, dim=1)
#         image_feature = self.image_pret_pos_encoder(image_feature)


#         video_feature = torch.mean(image_feature, dim=1)

#         print("video_feature", video_feature.size())
#         #image_feature = torch.cat([image_feature, v_num_face.unsqueeze(2).cuda()], dim=2)
#         print("image_feature", image_feature.size())
#         video_b,video_s,video_d = image_feature.size()

#         face_target = torch.zeros(video_b, video_s, self.hidden_size).cuda()
#         face_target[:, :, :1] = v_num_face.unsqueeze(2)

#         image_feature = self.image_modalspecific_transformer_model(image_feature, face_target.cuda())



        
#         print("video_d", video_d)
#         topic_distance_v = torch.from_numpy(self.k_means.transform(video_feature.detach().cpu().numpy())).cuda()

#         topic_distance_v_target = torch.zeros(video_b, self.hidden_size).cuda()
#         topic_distance_v_target[:, :self.topic_size] = topic_distance_v


#         image_feature = self.image_pret_transformer_model(image_feature, topic_distance_v_target.unsqueeze(1).expand(-1,video_s , -1))


#         v2t_attn, _ = self.v2tattn(image_feature, word_feature, word_feature)
#         v2t_attn = self.v2tattn_layer_norm(v2t_attn) + image_feature
#         v2t_attn_linear = self.v2tattn_linear(v2t_attn.reshape(-1, video_d))
#         v2t_attn = self.v2tattn_linear_layer_norm(v2t_attn_linear.view(video_b,video_s,video_d)) + v2t_attn

#         t2v_attn, _ = self.t2vattn(word_feature, image_feature, image_feature)
#         t2v_attn = self.v2tattn_layer_norm(t2v_attn) + word_feature
#         t2v_attn_linear = self.v2tattn_linear(t2v_attn.reshape(-1, text_d))
#         t2v_attn = self.v2tattn_linear_layer_norm(t2v_attn_linear.view(text_b,text_s,text_d)) + t2v_attn

#         print("v2t_attn", v2t_attn.size())
#         print("t2v_attn", t2v_attn.size())
#         #print("torch.cat([v2t_attn.squeeze(1), t2v_attn.squeeze(1)],dim=1)", torch.cat([v2t_attn.squeeze(1), t2v_attn.squeeze(1)],dim=1).size())
#         overall_feature = self.coattn(torch.cat([torch.mean(v2t_attn, dim=1), torch.mean(t2v_attn, dim=1)],dim=1)) 


#         #overall_feature = torch.mean(torch.stack([video_feature, sent_feature],dim=1), dim=1)

        
#         topic_distance = torch.from_numpy(self.k_means.transform(overall_feature.detach().cpu().numpy())).cuda()

#         topic_distance_target = torch.zeros(text_b, self.hidden_size).cuda()
#         topic_distance_target[:, :self.topic_size] = topic_distance


#         #word_feature_topic = torch.cat((word_feature, topic_distance.unsqueeze(1).expand(-1,text_s , -1)), dim=2)
#         #word_attn_output, _ = self.wordattn(word_feature.permute(1, 0, 2), topic_distance_target.unsqueeze(1).expand(-1,text_s , -1).permute(1, 0, 2), topic_distance_target.unsqueeze(1).expand(-1,text_s , -1).permute(1, 0, 2))
#         #word_attn_output = self.word_layer_norm(word_attn_output)
#         #print("word_attn_output", word_attn_output.size())
#         #word_attn_output = word_attn_output.permute(1, 0, 2)
#         #word_feature = self.word_pos_encoder(word_feature)
#         word_attn_output = self.word_transformer_model(word_feature, topic_distance_target.unsqueeze(1).expand(-1,text_s , -1))
#         #word_lstm_out = torch.cat([word_feature, word_attn_output.permute(1, 0, 2)], -1)
#         #print("word_lstm_out", word_lstm_out.size())
#         #word_lstm_out1, (_, _) = self.wordlstm(word_lstm_out)
#         #print("word_lstm_out1", word_lstm_out1.size())
#         print("word_attn_output", word_attn_output.size())

#         #text_logp = self.outputs2vocab(word_lstm_out1.reshape(-1, word_lstm_out1.size(2)))
#         text_logp = self.outputs2vocab(word_attn_output.reshape(-1, word_attn_output.size(2)))

#         #print("text_logp", text_logp.shape)
#         text_logp = text_logp.view(text_b, text_s, 1)


#         #text_b,text_s,_ = sent_feature.size()
#         #text_logp = self.outputs2vocab(sent_feature.reshape(-1, sent_feature.size(2)))
#         #print("text_logp", text_logp.shape)
#         #text_logp = text_logp.view(text_b, text_s, 1)
        
        
#         #image_feature_topic = torch.cat((image_feature, topic_distance.unsqueeze(1).expand(-1,video_s , -1)), dim=2)
#         #image_attn_output, _ = self.imageattn(image_feature.permute(1, 0, 2), topic_distance_target.unsqueeze(1).expand(-1,video_s , -1).permute(1, 0, 2), topic_distance_target.unsqueeze(1).expand(-1,video_s , -1).permute(1, 0, 2))
#         #image_attn_output = self.image_layer_norm(image_attn_output)
#         #image_attn_output = image_attn_output.permute(1, 0, 2)
#         #image_feature = self.image_pos_encoder(image_feature)
#         image_attn_output = self.image_transformer_model(image_feature, topic_distance_target.unsqueeze(1).expand(-1,video_s , -1))
#         #image_lstm_out = torch.cat([image_feature, image_attn_output.permute(1, 0, 2)], -1)
#         #image_lstm_out1, (_, _) = self.imagelstm(image_lstm_out)
#         print("image_attn_output", image_attn_output.size())

#         #video_logp = self.outputs2coverframe(image_lstm_out1.reshape(-1, image_lstm_out1.size(2)))
#         video_logp = self.outputs2coverframe(image_attn_output.reshape(-1, image_attn_output.size(2)))
#         #print("video_logp", video_logp.shape)

#         video_logp = video_logp.view(video_b, video_s, 1)
#         print("video_logp: ",video_logp)


#         output_video_summaries = []
#         output_video_summaries_pos = []

#         for image, summary in zip(scene_frame_pad_batch_list, video_logp):
#             #print("summary video", summary.size())
#             #output_video_summaries_pos.append(summary.argmax(dim=-1).item())
#             #rank = torch.topk(summary.squeeze(), self.max_summary_pic).indices
#             rank = torch.argsort(summary, dim=0, descending=True)
#             #print("rank video", rank[0])
#             # print("image", len(image))
#             output_video_summaries_pos.append(rank[0])
#             output_video_summaries.append(image[int(rank[0])])
        
#         #print("output_video_summaries_pos", output_video_summaries_pos)

#         output_text_summaries = []
#         output_text_summaries_pos = []

#         for text, t_id, summary in zip(text_token_list, text_id_list, text_logp):
#             word_count = 0
#             pos = []
#             #print("summary text", summary.size())
#             #print( "text", text)
#             #rank = torch.topk(summary.squeeze(), self.max_summary_word).indices
#             rank = torch.argsort(summary, dim=0, descending=True)
#             #print("rank text", rank)
#             text_id = []

#             filtered_rank = []
#             #for i in sorted(rank):
#             for i in rank:
#                 if i < len(text) and t_id[int(i)] < 49406:
#                     filtered_rank.append(int(i))
#                     if text[int(i)] != PAD_TOKEN and text[int(i)].isalnum():
#                         word_count += 1
                
#                 if word_count > self.max_summary_word:
#                     break

#             for i in sorted(filtered_rank):
#                 text_id.append(text[int(i)])
#                 pos.append(int(i))

#             #print("text_id", text_id)
#             output_text_summaries.append(" ".join(text_id))
#             output_text_summaries_pos.append(pos)


#         #for text, summary in zip(batch_sent_pad, text_logp):
#             #pos = []
#         #    print("summary text", summary)
#         #    print( "text", text)
#             #rank = torch.argsort(summary.squeeze(), descending=True)
#         #    rank = torch.argsort(summary, dim=0, descending=True)
#          #   print("rank text", rank)
#             #text_id = []

#             #for i in sorted(rank):
#             #    text_id.append(text[int(i)])
#             #    pos.append(int(i))
#             #print("text", text)
#             #print("summary", summary)
#             #print("rank", rank)
#           #  text_summary = ' '
#            # text_summary_pos = 0

#             #for rank_i in rank:
#              #   if text[rank_i] != ' ':
#               #      text_summary = text[rank_i]
#                #     text_summary_pos = rank_i
#                 #    break
                    
#             #output_text_summaries.append(text_summary)
#             #output_text_summaries_pos.append(text_summary_pos)
                    
#         print("output_text_summaries", output_text_summaries)
#         print("output_text_summaries_pos", output_text_summaries_pos)
#         print("output_video_summaries_pos",output_video_summaries_pos)
        
        
#         return output_text_summaries, output_text_summaries_pos, text_logp, output_video_summaries, output_video_summaries_pos, video_logp



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
        #self.cliptext.eval()
        #self.clipvision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.eval()
        self.cliptext.eval()
        #self.clipvision.eval()
        self.hidden_size = 512
        self.topic_size = 100
        self.dropout = 0.1
        #self.image_hidden_size=768
        self.num_attention_head = 16
        #self.wordattn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_head, dropout=self.dropout)
        #self.word_layer_norm = nn.LayerNorm(self.hidden_size)
        #self.wordlstm= nn.LSTM(self.hidden_size*2, self.hidden_size, num_layers=2, bidirectional=False, batch_first = True)

        self.word_pret_pos_encoder = PositionalEncoding(self.hidden_size)
        #word_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16)
        #self.word_pret_transformer_model = nn.TransformerEncoder(word_encoder_layer, num_layers=12)
        self.word_pret_transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)

        self.word_pos_encoder = PositionalEncoding(self.hidden_size)
        self.word_transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)

        self.image_pret_pos_encoder = PositionalEncoding(self.hidden_size)
        #image_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16)
        #self.image_pret_transformer_model = nn.TransformerEncoder(image_encoder_layer, num_layers=12)
        self.image_pret_transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)

        self.image_pos_encoder = PositionalEncoding(self.hidden_size)
        self.image_transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)


        self.v2tattn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_head, batch_first=True, dropout=self.dropout)
        self.v2tattn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.v2tattn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v2tattn_linear_layer_norm = nn.LayerNorm(self.hidden_size)
        
        self.t2vattn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_head, batch_first=True, dropout=self.dropout)
        self.t2vattn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.t2vattnn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.t2vattn_linear_layer_norm = nn.LayerNorm(self.hidden_size)

        self.coattn = nn.Linear(self.hidden_size*2, self.hidden_size)
        #self.image_layer_norm = nn.LayerNorm(self.hidden_size)
        #self.imagelstm= nn.LSTM(self.hidden_size*2, self.hidden_size, num_layers=2, bidirectional=False, batch_first = True)
        

        with open("kmeans_model_100.pkl", "rb") as f:
            self.k_means =  pkl.load(f)

        self.outputs2vocab = nn.Linear(self.hidden_size , 1)

       # self.pic_linear_z = nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size)

        self.outputs2coverframe = nn.Linear(self.hidden_size, 1)
        

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

        batch_sent_pad_t = [list(x) for x in zip(*batch_sent_pad)]
        sent_feature = []
        word_feature = []
        pad_sent_len = []
        sent_feature = []
        last_len = 0
        #for sent in batch_sent_pad_t:
        #print("sent", sent)
        text_id_list = torch.nn.utils.rnn.pad_sequence(text_id_list, batch_first=True).cuda()
        print("text_id_list", text_id_list.size())
        word_feature = self.cliptext(text_id_list).last_hidden_state
        word_feature = self.word_pret_pos_encoder(word_feature)
        #word_feature = self.word_pret_transformer_model(word_feature.permute(1,0,2)).permute(1,0,2)
        text_b,text_s,text_d = word_feature.size()

        print("word_feature", word_feature.size())
        #text_overall_feature = self.clip.get_text_features(text_token['input_ids'].cuda())
        #text_token_list.append(text_token_l)
        sent_feature = self.clip.get_text_features(text_id_list)
        print("sent_feature", sent_feature.size())
        
        topic_distance_t = torch.from_numpy(self.k_means.transform(sent_feature.detach().cpu().numpy())).cuda()

        topic_distance_t_target = torch.zeros(text_b, 512).cuda()
        topic_distance_t_target[:, :self.topic_size] = topic_distance_t

        word_feature = self.word_pret_transformer_model(word_feature, topic_distance_t_target.unsqueeze(1).expand(-1,text_s , -1))

        last_len = 0

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
        print("image_feature", image_feature.size())

        video_feature = torch.mean(image_feature, dim=1)

        print("video_feature", video_feature.size())

        video_b,video_s,video_d = image_feature.size()

        topic_distance_v = torch.from_numpy(self.k_means.transform(video_feature.detach().cpu().numpy())).cuda()

        topic_distance_v_target = torch.zeros(video_b, 512).cuda()
        topic_distance_v_target[:, :self.topic_size] = topic_distance_v


        image_feature = self.image_pret_transformer_model(image_feature, topic_distance_v_target.unsqueeze(1).expand(-1,video_s , -1))


        v2t_attn, _ = self.v2tattn(image_feature, word_feature, word_feature)
        v2t_attn = self.v2tattn_layer_norm(v2t_attn) + image_feature
        v2t_attn_linear = self.v2tattn_linear(v2t_attn.reshape(-1, video_d))
        v2t_attn = self.v2tattn_linear_layer_norm(v2t_attn_linear.view(video_b,video_s,video_d)) + v2t_attn

        t2v_attn, _ = self.t2vattn(word_feature, image_feature, image_feature)
        t2v_attn = self.v2tattn_layer_norm(t2v_attn) + word_feature
        t2v_attn_linear = self.v2tattn_linear(t2v_attn.reshape(-1, text_d))
        t2v_attn = self.v2tattn_linear_layer_norm(t2v_attn_linear.view(text_b,text_s,text_d)) + t2v_attn

        print("v2t_attn", v2t_attn.size())
        print("t2v_attn", t2v_attn.size())
        #print("torch.cat([v2t_attn.squeeze(1), t2v_attn.squeeze(1)],dim=1)", torch.cat([v2t_attn.squeeze(1), t2v_attn.squeeze(1)],dim=1).size())
        overall_feature = self.coattn(torch.cat([torch.mean(v2t_attn, dim=1), torch.mean(t2v_attn, dim=1)],dim=1)) 


        #overall_feature = torch.mean(torch.stack([video_feature, sent_feature],dim=1), dim=1)

        
        topic_distance = torch.from_numpy(self.k_means.transform(overall_feature.detach().cpu().numpy())).cuda()

        topic_distance_target = torch.zeros(text_b, 512).cuda()
        topic_distance_target[:, :self.topic_size] = topic_distance


        #word_feature_topic = torch.cat((word_feature, topic_distance.unsqueeze(1).expand(-1,text_s , -1)), dim=2)
        #word_attn_output, _ = self.wordattn(word_feature.permute(1, 0, 2), topic_distance_target.unsqueeze(1).expand(-1,text_s , -1).permute(1, 0, 2), topic_distance_target.unsqueeze(1).expand(-1,text_s , -1).permute(1, 0, 2))
        #word_attn_output = self.word_layer_norm(word_attn_output)
        #print("word_attn_output", word_attn_output.size())
        #word_attn_output = word_attn_output.permute(1, 0, 2)
        word_feature = self.word_pos_encoder(word_feature)
        word_attn_output = self.word_transformer_model(word_feature, topic_distance_target.unsqueeze(1).expand(-1,text_s , -1))
        #word_lstm_out = torch.cat([word_feature, word_attn_output.permute(1, 0, 2)], -1)
        #print("word_lstm_out", word_lstm_out.size())
        #word_lstm_out1, (_, _) = self.wordlstm(word_lstm_out)
        #print("word_lstm_out1", word_lstm_out1.size())
        print("word_attn_output", word_attn_output.size())

        #text_logp = self.outputs2vocab(word_lstm_out1.reshape(-1, word_lstm_out1.size(2)))
        text_logp = self.outputs2vocab(word_attn_output.reshape(-1, word_attn_output.size(2)))

        #print("text_logp", text_logp.shape)
        text_logp = text_logp.view(text_b, text_s, 1)


        #text_b,text_s,_ = sent_feature.size()
        #text_logp = self.outputs2vocab(sent_feature.reshape(-1, sent_feature.size(2)))
        #print("text_logp", text_logp.shape)
        #text_logp = text_logp.view(text_b, text_s, 1)
        
        
        #image_feature_topic = torch.cat((image_feature, topic_distance.unsqueeze(1).expand(-1,video_s , -1)), dim=2)
        #image_attn_output, _ = self.imageattn(image_feature.permute(1, 0, 2), topic_distance_target.unsqueeze(1).expand(-1,video_s , -1).permute(1, 0, 2), topic_distance_target.unsqueeze(1).expand(-1,video_s , -1).permute(1, 0, 2))
        #image_attn_output = self.image_layer_norm(image_attn_output)
        #image_attn_output = image_attn_output.permute(1, 0, 2)
        image_feature = self.image_pos_encoder(image_feature)
        image_attn_output = self.image_transformer_model(image_feature, topic_distance_target.unsqueeze(1).expand(-1,video_s , -1))
        #image_lstm_out = torch.cat([image_feature, image_attn_output.permute(1, 0, 2)], -1)
        #image_lstm_out1, (_, _) = self.imagelstm(image_lstm_out)
        print("image_attn_output", image_attn_output.size())

        #video_logp = self.outputs2coverframe(image_lstm_out1.reshape(-1, image_lstm_out1.size(2)))
        video_logp = self.outputs2coverframe(image_attn_output.reshape(-1, image_attn_output.size(2)))
        #print("video_logp", video_logp.shape)

        video_logp = video_logp.view(video_b, video_s, 1)
        print("video_logp: ",video_logp.size())


        output_video_summaries = []
        output_video_summaries_pos = []

        for image, summary in zip(scene_frame_pad_batch_list, video_logp):
            #print("summary video", summary.size())
            #output_video_summaries_pos.append(summary.argmax(dim=-1).item())
            #rank = torch.topk(summary.squeeze(), self.max_summary_pic).indices
            rank = torch.argsort(summary, dim=0, descending=True)
            #print("rank video", rank[0])
            print("image", len(image))
            output_video_summaries_pos.append(rank[0])
            output_video_summaries.append(image[int(rank[0])])
        
        #print("output_video_summaries_pos", output_video_summaries_pos)

        output_text_summaries = []
        output_text_summaries_pos = []

        for text, t_id, summary in zip(text_token_list, text_id_list, text_logp):
            word_count = 0
            pos = []
            #print("summary text", summary.size())
            #print( "text", text)
            #rank = torch.topk(summary.squeeze(), self.max_summary_word).indices
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


        #for text, summary in zip(batch_sent_pad, text_logp):
            #pos = []
        #    print("summary text", summary)
        #    print( "text", text)
            #rank = torch.argsort(summary.squeeze(), descending=True)
        #    rank = torch.argsort(summary, dim=0, descending=True)
         #   print("rank text", rank)
            #text_id = []

            #for i in sorted(rank):
            #    text_id.append(text[int(i)])
            #    pos.append(int(i))
            #print("text", text)
            #print("summary", summary)
            #print("rank", rank)
          #  text_summary = ' '
           # text_summary_pos = 0

            #for rank_i in rank:
             #   if text[rank_i] != ' ':
              #      text_summary = text[rank_i]
               #     text_summary_pos = rank_i
                #    break
                    
            #output_text_summaries.append(text_summary)
            #output_text_summaries_pos.append(text_summary_pos)
                    
        print("output_text_summaries", output_text_summaries)
        print("output_text_summaries_pos", output_text_summaries_pos)
        print("output_video_summaries_pos",output_video_summaries_pos)
        
        
        return output_text_summaries, output_text_summaries_pos, text_logp, output_video_summaries, output_video_summaries_pos, video_logp
