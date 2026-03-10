import torch
import torch.nn as nn
import torch.nn.functional as F
from .trm import *
import pandas as pd
import json
from .attention import *
from torch.nn.parameter import Parameter
import copy
import math
# BEST
class D_query_frame(torch.nn.Module):
    def __init__(self,dataset):
        super(D_query_frame,self).__init__()
        self.modelA = SGAN(dataset=dataset)

    def forward(self,  **kwargs):
        vid = kwargs['vid']
        label = kwargs['label']
        fps = kwargs['fps']
        total_frame = kwargs['total_frame']
        all_phrase_semantic_fea = kwargs['all_phrase_semantic_fea']
        all_phrase_emo_fea = kwargs['all_phrase_emo_fea']
        raw_visual_frames = kwargs['raw_visual_frames']
        raw_audio_emo = kwargs['raw_audio_emo']
        ocr_pattern_fea = kwargs['ocr_pattern_fea']
        ocr_phrase_fea = kwargs['ocr_phrase_fea']
        ocr_time_region = kwargs['ocr_time_region']
        visual_frames_fea = kwargs['visual_frames_fea']
        visual_frames_seg_indicator = kwargs['visual_frames_seg_indicator']
        visual_seg_paded = kwargs['visual_seg_paded']

        output = self.modelA(all_phrase_semantic_fea, all_phrase_emo_fea, raw_visual_frames, raw_audio_emo)
        return output

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class VerificationModule(nn.Module):
    def __init__(self, d_model, img2text_attn_args=None, img_query_with_pos=True,
                 img2textcond_attn_args=None, img2img_attn_args=None, vl_verify=None):
        super().__init__()

        self.img2text_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first= True) # MultiheadAttention
        self.img_query_with_pos = img_query_with_pos

        self.text_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=2)
        self.img_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
        self.tf_pow = 2.0
        self.tf_scale = Parameter(torch.Tensor([1.0]))
        self.tf_sigma = Parameter(torch.Tensor([0.5]))

        self.img2textcond_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first= True) # MultiheadAttention

        self.img2img_attn =  nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first= True)

        self.norm_text_cond_img = nn.LayerNorm(d_model)
        self.norm_img = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, word_feat, word_pos = None):
        orig_img_feat = img_feat

        # visual-linguistic verification
        img_query = self.with_pos_embed(img_feat, word_pos)
        text_info = self.img2text_attn(
            query=img_query, key=word_feat,
            value=word_feat)[0]

        text_embed = self.text_proj(text_info)
        img_embed = self.img_proj(img_feat)
        verify_score = (F.normalize(img_embed, p=2, dim=-1) *
                        F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        verify_score = self.tf_scale * \
                       torch.exp( - (1 - verify_score).pow(self.tf_pow) \
                        / (2 * self.tf_sigma**2))

        # language-guided context encoder
        text_cond_info = self.img2textcond_attn(
            query=img_feat, key= word_feat,
            value=word_feat)[0]

        q = k = img_feat + text_cond_info
        text_cond_img_ctx = self.img2img_attn(
            query=q, key=k, value=img_feat)[0]

        # discriminative feature
        fuse_img_feat = (self.norm_img(img_feat) +
                         self.norm_text_cond_img(text_cond_img_ctx)) * verify_score

        return orig_img_feat+fuse_img_feat

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SGAN(torch.nn.Module):
    def __init__(self, dataset, num_extra_layers=1):
        if dataset=='fakett':
            self.encoded_text_semantic_fea_dim=512
        elif dataset=='fakesv':
            self.encoded_text_semantic_fea_dim=768
        self.input_visual_frames=83
        super().__init__()

        self.mlp_text_emo = nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_text_semantic = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim,256),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_img = nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_audio = nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(),nn.Dropout(0.1))

        self.extra_encoder_layer = VerificationModule(256)
        self.learnable_embed_frames = nn.Embedding(83, 256)    #     num_visual_frames=83, num_segs=83, num_phrase=80
        self.learnable_embed_phrase = nn.Embedding(512, 256)

        self.vis_query_embed = nn.Embedding(1, 256)
        self.slf_query_embed = nn.Embedding(1, 256)
        self.emo_query_embed = nn.Embedding(1, 256)
        self.aud_query_embed = nn.Embedding(1, 256)

        self.bounding_a = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first= True)
        self.bounding_b = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first= True)
        self.bounding_c = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first= True)
        self.bounding_d = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first= True)


        self.filter1 = nn.Sequential(
            nn.Conv1d(4, 4, kernel_size=1),
            nn.LayerNorm(512, eps=1e-3),
            nn.ReLU(inplace=True))

        self.filter2 = nn.Conv1d(512, 256, kernel_size=1)

        self.ffn = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.1),
                                 nn.Linear(512, 2))

        self.norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, raw_semantic_fea, raw_emo_fea, raw_visual_frames, raw_audio_emo):

        bs, _, __ = raw_semantic_fea.shape

        raw_t_fea_semantic=self.mlp_text_semantic(raw_semantic_fea)  # 语义 2 512 256
        raw_t_fea_emo=self.mlp_text_emo(raw_emo_fea).unsqueeze(1)      # 情感 2 1 256
        raw_v_fea=self.mlp_img(raw_visual_frames)                           # 视觉 2 83 256
        raw_a_fea_emo=self.mlp_audio(raw_audio_emo).unsqueeze(1)            # 音频 2 1 256

        # Encode discriminative features
        pos_frames = self.learnable_embed_frames.weight.unsqueeze(0).repeat(bs, 1, 1)
        pos_phrase = self.learnable_embed_phrase.weight.unsqueeze(0).repeat(bs, 1, 1)

        semantic_emo = self.extra_encoder_layer(raw_t_fea_emo,raw_t_fea_semantic)  # 4,1,256
        semantic_aud = self.extra_encoder_layer(raw_a_fea_emo,raw_t_fea_semantic)  #
        semantic_vis = self.extra_encoder_layer(raw_v_fea,raw_t_fea_semantic, pos_frames)
        semantic_slf = self.extra_encoder_layer(raw_t_fea_semantic,raw_t_fea_semantic, pos_phrase)

        vis_query_embed = self.vis_query_embed.weight.unsqueeze(1).repeat(bs, 1, 1)
        slf_query_embed = self.slf_query_embed.weight.unsqueeze(1).repeat(bs, 1, 1)
        emo_query_embed = self.emo_query_embed.weight.unsqueeze(1).repeat(bs, 1, 1)
        aud_query_embed = self.aud_query_embed.weight.unsqueeze(1).repeat(bs, 1, 1)

        # Flexible Query Learning
        vis_query = torch.zeros_like(vis_query_embed)
        slf_query = torch.zeros_like(slf_query_embed)
        emo_query = torch.zeros_like(emo_query_embed)
        aud_query = torch.zeros_like(aud_query_embed)


        a = self.bounding_a(query=vis_query+vis_query_embed,
                                   key=semantic_vis,
                                   value=raw_v_fea)[0]

        b = self.bounding_b(query=slf_query+slf_query_embed,
                                   key=semantic_slf,
                                   value=raw_t_fea_semantic)[0]

        c = self.bounding_c(query=emo_query+emo_query_embed,
                                   key=semantic_emo,
                                   value=raw_t_fea_emo)[0]

        d = self.bounding_d(query=aud_query+aud_query_embed,
                                   key=semantic_aud,
                                   value=raw_a_fea_emo)[0]



        vis_conv = torch.cat([a, vis_query], dim = -1)
        slf_conv = torch.cat([b, slf_query], dim = -1)
        emo_conv = torch.cat([c, emo_query], dim = -1)
        aud_conv = torch.cat([d, aud_query], dim = -1)

        # Modality Integration
        dis_bound = torch.cat([vis_conv,slf_conv,emo_conv,aud_conv],dim = -2)
        dis_bounding = self.filter1(dis_bound)
        dis_bounding_transposed = dis_bounding.transpose(1, 2)
        dis_bounding = self.filter2(dis_bounding_transposed).transpose(1, 2)

        dis_splitted = torch.split(dis_bounding, 1, dim =1)

        bounding_query = torch.cat([dis_splitted[0]+vis_query,dis_splitted[1]+slf_query,dis_splitted[2]+emo_query,dis_splitted[3]+aud_query],dim = -1).squeeze(1)
        bounding_query = self.ffn(self.norm(self.dropout(bounding_query)))

        return bounding_query
