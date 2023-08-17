import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from collections import OrderedDict
from transformers import BertModel, BertConfig, BertTokenizer
from gesture_word_prediction.model import GestureWordPredictor

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, dropout=0, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.Dropout(p=dropout),
                    nn.BatchNorm1d(out_features), 
                    nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), 
                    nn.Dropout(p=dropout),
                    nonlinearity)
    
    def forward(self, x):
        return self.model(x)


class conv(nn.Module):
    def __init__(self, nin, nout):
        super(conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )
    
    def forward(self, input):
        return self.main(input)


class gesture_encoder(nn.Module):
    def __init__(self, opt):
        super(gesture_encoder, self).__init__()
        self.hidden_dim = opt.rnn_size
        self.frames = opt.frames
        self.g_dim = opt.g_dim
        self.z_dim = opt.z_dim

        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.g_linear = LinearUnit(self.hidden_dim * 2, self.z_dim, False, dropout=opt.dropout)
    
    def forward(self, x_gesture):
        x = x_gesture.reshape([x_gesture.shape[0], x_gesture.shape[1], -1])     # [32bs, 8frame, 24dim]
        lstm_out, _ = self.z_lstm(x)       # lstm_out: [128, 30, 512]
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]  # backward: [128, 256]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]       # frontal: [128, 256]
        lstm_out_f = torch.cat((frontal, backward), dim=1)          # lstm_out_f: [128, 512]
        ges_f = self.g_linear(lstm_out_f)
        return ges_f


class gesture_decoder(nn.Module):
    def __init__(self, opt):
        super(gesture_decoder, self).__init__()
        self.channels = 3
        self.z_dim = opt.z_dim
        self.g_dim = opt.g_dim
        self.n_joint = int(opt.g_dim / self.channels)
        self.frames = opt.frames
        # self.linear = nn.Linear(self.z_dim, self.frames*self.g_dim)
        self.linear = LinearUnit(self.z_dim, self.frames*self.g_dim, False, dropout=opt.dropout)
        self.linear_kf = LinearUnit(self.z_dim * 2, self.frames+1, False, dropout=opt.dropout)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, text_f_kf):   # input: ([32, 8, 32])  =  (batch_size, frame, feature)
        out = self.linear(input)    # [32, 168]
        out = out.view(out.shape[0], self.frames, self.g_dim)   # [32, 8, 21]
        output = out.view(out.shape[0], self.frames, self.n_joint, self.channels)   # [32, 8, 7, 3]

        # return output
        input_kf = torch.concat([input, text_f_kf], axis=1)
        out_kf = self.linear_kf(input_kf)
        out_kf = self.sm(out_kf)

        return output, out_kf 


class text_encoder(nn.Module):
    def __init__(self, opt):
        super(text_encoder, self).__init__()
        self.input_dim = opt.text_length * opt.t_dim
        self.z_dim = opt.z_dim  # motion (dynamic)
        self.hidden_dim = 256
        self.batch_size = opt.batch_size

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.z_dim)
        self.linear1_kf = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2_kf = nn.Linear(self.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x_text):
        t = x_text.view(self.batch_size, -1)
        text_f = self.dropout(self.linear2(self.linear1(t)))
        text_f_kf = self.dropout(self.linear2_kf(self.linear1_kf(t)))

        return text_f, text_f_kf

    
class text_decoder(nn.Module):
    def __init__(self, opt, bidirectional=True):
        super(text_decoder, self).__init__()
        self.embedding_size = BertConfig.from_pretrained(opt.bert_kind).hidden_size
        self.vocab_size = BertConfig.from_pretrained(opt.bert_kind).vocab_size
        self.hidden_factor = (2 if bidirectional else 1) * opt.f_rnn_layers
        self.latent_size = opt.z_dim
        self.hidden_size = opt.rnn_size # 256
        self.text_length = opt.text_length
        self.batch_size = opt.batch_size

        self.decoder_lstm = nn.LSTM(self.hidden_size, self.hidden_size, opt.f_rnn_layers, bidirectional=True, batch_first=True)
        self.latent2hidden = nn.Linear(self.latent_size, self.text_length * self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size * (2 if bidirectional else 1), self.vocab_size)

    def forward(self, z):   # input: ([32, 8, 32])  =  (batch_size, frame, feature)
        hidden = self.latent2hidden(z)
        hidden = hidden.view(self.batch_size, self.text_length, self.hidden_size)
        output, hidden = self.decoder_lstm(hidden)
        logp = F.softmax(self.outputs2vocab(output), dim=-1)

        return logp


class ACT2G(nn.Module):
    def __init__(self, opt):
        super(ACT2G, self).__init__()
        self.opt = opt
        
        if opt.wo_attn:
            options_name = "bert-base-uncased"
            embed_dim = BertConfig.from_pretrained(options_name).hidden_size
            self.bert_model = BertModel.from_pretrained(options_name)
            self.linear = nn.Linear(embed_dim, opt.t_dim)

            # Freeze BERT
            for param in self.bert_model.parameters():
                param.requires_grad = False
        else:
            gwp_saved_model = torch.load(opt.gwp_model)
            self.gwp = GestureWordPredictor(gwp_saved_model['option'])
            if opt.mode == 'train':
                self.gwp.load_state_dict(gwp_saved_model['model'])
        
        self.gesture_encoder = gesture_encoder(opt)
        self.gesture_decoder = gesture_decoder(opt)
        self.text_encoder = text_encoder(opt)

        # # Freeze Encoder 
        # for param in self.gesture_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

        # # Freeze Decoder 
        # for param in self.gesture_decoder.parameters():
        #     param.requires_grad = False

    def forward(self, x_gesture, x_text):
        gesture_feature = self.gesture_encoder(x_gesture)

        if self.opt.wo_attn == 1:
            embed, _, = self.bert_model(x_text.type(torch.LongTensor).cuda())
            gwp_out = self.linear(embed)
            attn = torch.zeros([self.opt.batch_size, x_text.shape[1]])  # dummy
        else:
            gwp_out, attn = self.gwp(x_text)

        text_feature, text_f_kf = self.text_encoder(gwp_out)

        recon_g, kf_pred = self.gesture_decoder(gesture_feature, text_f_kf)
        return recon_g, gesture_feature, text_feature, attn, kf_pred

    def generate_gesture(self, x_text, attention=None):
        if self.opt.wo_attn == 1:
            embed, _, = self.bert_model(x_text.type(torch.LongTensor).cuda())
            gwp_out = self.linear(embed)
            attn = torch.zeros([self.opt.batch_size, x_text.shape[1]])  # dummy
        else:
            gwp_out, attn = self.gwp(x_text, attention)
        text_feature, text_f_kf  = self.text_encoder(gwp_out)

        recon_g, kf_pred = self.gesture_decoder(text_feature, text_f_kf)
        return recon_g, text_feature, attn, kf_pred

        
    


class gesture_encoder_vae(nn.Module):
    def __init__(self, opt):
        super(gesture_encoder_vae, self).__init__()
        self.hidden_dim = opt.rnn_size
        self.frames = opt.frames
        self.g_dim = opt.g_dim
        self.z_dim = opt.z_dim

        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.g_mean = LinearUnit(self.hidden_dim * 2, self.z_dim, False)
        self.g_logvar = LinearUnit(self.hidden_dim * 2, self.z_dim, False)
    
    def forward(self, x_gesture):
        x = x_gesture.reshape([x_gesture.shape[0], x_gesture.shape[1], -1])     # [32bs, 8frame, 24dim]
        lstm_out, _ = self.z_lstm(x)       # lstm_out: [128, 30, 512]
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]  # backward: [128, 256]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]       # frontal: [128, 256]
        lstm_out_f = torch.cat((frontal, backward), dim=1)          # lstm_out_f: [128, 512]
        g_mean = self.g_mean(lstm_out_f)
        g_logvar = self.g_logvar(lstm_out_f)
        return g_mean, g_logvar


class gesture_decoder_linear(nn.Module):
    def __init__(self, opt):
        super(gesture_decoder_linear, self).__init__()
        self.channels = 3
        self.z_dim = opt.z_dim
        self.g_dim = opt.g_dim
        self.n_joint = int(opt.g_dim / self.channels)
        self.frames = opt.frames
        self.hidden_dim = 32
        self.linear = nn.Linear(self.z_dim, self.frames*self.g_dim)

    def forward(self, input):   # input: ([32, 8, 32])  =  (batch_size, frame, feature)
        out = self.linear(input)    # [32, 168]
        out = out.view(out.shape[0], self.frames, self.g_dim)   # [32, 8, 21]
        output = out.view(out.shape[0], self.frames, self.n_joint, self.channels)   # [32, 8, 7, 3]
        return output 


class GVAE(nn.Module):
    def __init__(self, opt):
        super(GVAE, self).__init__()
        self.gesture_encoder = gesture_encoder_vae(opt)
        # self.gesture_decoder = gesture_decoder_vae(opt)
        self.gesture_decoder = gesture_decoder_linear(opt)


    def forward(self, x_gesture):
        g_mean, g_logvar = self.gesture_encoder(x_gesture)

        g_post = self.reparameterize(g_mean, g_logvar, random_sampling=True)

        recon_g = self.gesture_decoder(g_post)

        return recon_g, g_mean, g_post, g_logvar

    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean