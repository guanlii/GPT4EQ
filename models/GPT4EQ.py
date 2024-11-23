import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_time

from ._factory import register_model


class ConvBlock(nn.Module):
    """Convolution block
    Input shape: (N,C,L)
    """

    _epsilon = 1e-6

    def __init__(
        self, in_channels, out_channels, kernel_size, kernel_l1_alpha, bias_l1_alpha
    ):
        super().__init__()

        assert kernel_l1_alpha >= 0.0
        assert bias_l1_alpha >= 0.0

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, padding=0)

        if kernel_l1_alpha > 0.0:
            self.conv.weight.register_hook(
                lambda grad: grad.data
                + kernel_l1_alpha * torch.sign(self.conv.weight.data)
            )
        if bias_l1_alpha > 0.0:
            self.conv.bias.register_hook(
                lambda grad: grad.data + bias_l1_alpha * torch.sign(self.conv.bias.data)
            )

    def forward(self, x):
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv(x)
        x = self.relu(x)
        x = F.pad(x, (0, x.size(-1) % 2), "constant", -1 / self._epsilon)
        x = self.pool(x)
        #print(x.shape)
        return x

class ResConvBlock(nn.Module):
    """Residual convolution block
    Input shape: (N,C,L)
    """

    def __init__(self, io_channels, kernel_size, drop_rate):
        super().__init__()

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.bn0 = nn.BatchNorm1d(num_features=io_channels)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout1d(p=drop_rate)
        self.conv0 = nn.Conv1d(
            in_channels=io_channels, out_channels=io_channels, kernel_size=kernel_size
        )

        self.bn1 = nn.BatchNorm1d(io_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(p=drop_rate)
        self.conv1 = nn.Conv1d(
            in_channels=io_channels, out_channels=io_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        x1 = self.bn0(x)
        x1 = self.relu0(x1)
        x1 = self.dropout0(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv0(x1)

        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv1(x1)
        out = x + x1
        #print(out.shape)
        return out

class BiLSTMBlock(nn.Module):
    """Bi-LSTM block
    Input shape: (N,C,L)
    """

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=out_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=drop_rate)
        self.conv = nn.Conv1d(
            in_channels=2 * out_channels, out_channels=out_channels, kernel_size=1
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        #print(x.shape)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        out_samples,
        kernel_size,
        kernel_l1_alpha,
        bias_l1_alpha,
    ):
        super().__init__()

        assert kernel_l1_alpha >= 0.0
        assert bias_l1_alpha >= 0.0

        self.out_samples = out_samples

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()

        if kernel_l1_alpha > 0.0:
            self.conv.weight.register_hook(
                lambda grad: grad.data
                + kernel_l1_alpha * torch.sign(self.conv.weight.data)
            )
        if bias_l1_alpha > 0.0:
            self.conv.bias.register_hook(
                lambda grad: grad.data + bias_l1_alpha * torch.sign(self.conv.bias.data)
            )

    def forward(self, x):
        x = self.upsampling(x)
        x = x[:, :, : self.out_samples]
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        #print(x.shape)
        x = self.conv(x)
        x = self.relu(x)

        return x

class IdentityNTuple(nn.Identity):
    def __init__(self, *args, ntuple: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        assert ntuple >= 1
        self.ntuple = ntuple

    def forward(self, input: torch.Tensor):
        if self.ntuple > 1:
            return (super().forward(input),) * self.ntuple
        else:
            return super().forward(input)

class Decoder(nn.Module):
    """
    Decoder layers:
        LSTM * 1 (opt.)
        TransormerLayer *1 (opt.)
        UpSampling * 7
        Conv * 1
    """

    def __init__(
        self,
        conv_channels: list,
        conv_kernels: list,
        transformer_io_channels: int,
        drop_rate: float,
        out_samples,
        has_lstm: bool = True,
        has_local_attn: bool = True,
        local_attn_width: int = 3,
        conv_kernel_l1_regularization: float = 0.0,
        conv_bias_l1_regularization: float = 0.0,
    ):
        super().__init__()

        self.lstm = (
            nn.LSTM(
                input_size=transformer_io_channels,
                hidden_size=transformer_io_channels,
                batch_first=True,
                bidirectional=False,
            )
            if has_lstm
            else IdentityNTuple(ntuple=2)
        )

        self.lstm_dropout = nn.Dropout(p=drop_rate) if has_lstm else nn.Identity()



        crop_sizes = [out_samples]
        for _ in range(len(conv_kernels) - 1):
            crop_sizes.insert(0, math.ceil(crop_sizes[0] / 2))

        self.upsamplings = nn.Sequential(
            *[
                UpSamplingBlock(
                    in_channels=inc,
                    out_channels=outc,
                    out_samples=crop,
                    kernel_size=kers,
                    kernel_l1_alpha=conv_kernel_l1_regularization,
                    bias_l1_alpha=conv_bias_l1_regularization,
                )
                for inc, outc, crop, kers in zip(
                    [transformer_io_channels] + conv_channels[:-1],
                    conv_channels,
                    crop_sizes,
                    conv_kernels,
                )
            ]
        )

        self.conv_out = nn.Conv1d(
            in_channels=conv_channels[-1], out_channels=1, kernel_size=11, padding=5
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)

        x = self.upsamplings(x)

        x = self.conv_out(x)

        x = x.sigmoid()

        return x

class Model(nn.Module):
    
    def __init__(self,
                 in_channels=3,
                 in_samples=6000,
                 ln=False,
                 pred_len=0,
                 seq_len=100,
                 patch_size=1,
                 stride=1,
                 d_ff=8,
                 enc_in=3,
                 c_out=3,
                 d_model=768,
                 embed='timeF',
                 freq='h',
                 dropout=0.1,
                 drop_rate=0.1,
                 gpt_layers=6,
#                  bert_layers=1,
                 pretrain=True,
                 use_gpu=True,
                 conv_kernel_l1_regularization: float = 0.0,
                 conv_bias_l1_regularization: float = 0.0,
                 conv_channels=[8, 16, 32],
                 conv_kernels=[11, 9, 7],
                 resconv_kernels=[],
                 num_lstm_blocks=0,
                 transformer_io_channels=16,
                 decoder_with_attn_lstm=[False, True, True],
                 **kwargs
                ):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.transformer_io_channels = transformer_io_channels
        self.drop_rate=drop_rate
        self.is_ln = ln
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_ff = d_ff
        self.patch_num = (self.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(enc_in * self.patch_size, d_model, embed, freq,
                                           dropout)
        self.gpt_layers = gpt_layers
#         self.bert_layers = bert_layers
        self.pretrain = pretrain
        

        #self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        
        #self.gpt2.h = self.gpt2.h[:gpt_layers]
        #是否用预训练权重
        if pretrain:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        #print("gpt2 = {}".format(self.gpt2))
        
#         # Use BERT model instead of GPT-2
#         if pretrain:
#             self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True, output_hidden_states=True)
#         else:
#             print("------------------no pretrain------------------")
#             self.bert = BertModel._from_config(BertConfig())

        self.decoder_with_attn_lstm = decoder_with_attn_lstm
        
        
        #冻结FFN、Atten，只微调归一化层和位置嵌入层
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: 
                param.requires_grad = True
            else:
                param.requires_grad = False
        #全量微调
#         for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#             param.requires_grad = True
        #冻结Atten
#         for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#             if 'ln' in name or 'wpe' in name or 'mlp'in name: 
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
        #冻结FFN
#         for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#             if 'ln' in name or 'wpe' in name or 'attn'in name: 
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
        #只调layernorm
#         for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#             if 'ln' in name: 
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
        
        
        #冻结所有gpt层
#         for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#             param.requires_grad = False

        if use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)
#             self.bert.to(device=device)

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)

        # Conv 1D
        self.convs = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_size=kers,
                    kernel_l1_alpha=conv_kernel_l1_regularization,
                    bias_l1_alpha=conv_bias_l1_regularization,
                )
                for inc, outc, kers in zip(
                    [in_channels] + conv_channels[:-1], conv_channels, conv_kernels
                )
            ]
        )

        # Res CNN
        self.res_convs = nn.Sequential(
            *[
                ResConvBlock(
                    io_channels=conv_channels[-1], kernel_size=kers, drop_rate=drop_rate
                )
                for kers in resconv_kernels
            ]
        )

        # Bi-LSTM
        self.bilstms = nn.Sequential(
            *[
                BiLSTMBlock(in_channels=inc, out_channels=outc, drop_rate=drop_rate)
                for inc, outc in zip(
                    [conv_channels[-1]]
                    + [transformer_io_channels] * (num_lstm_blocks - 1),
                    [transformer_io_channels] * num_lstm_blocks,
                )
            ]
        )

        self.ln_proj = nn.LayerNorm(d_ff)
        self.out_layer = nn.Linear(
            d_ff, 
            16, 
            bias=True)
        self.sigmoid = nn.Sigmoid()
        
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    conv_channels=self.conv_channels[::-1],
                    conv_kernels=self.conv_kernels[::-1],
                    transformer_io_channels=self.transformer_io_channels,
                    drop_rate=self.drop_rate,
                    out_samples=self.in_samples,
                    has_lstm=has_attn_lstm,
                    conv_kernel_l1_regularization=conv_kernel_l1_regularization,
                    conv_bias_l1_regularization=conv_bias_l1_regularization,
                )
                for has_attn_lstm in self.decoder_with_attn_lstm
            ]
        )
        

    def forward(self, x_enc, mask=None):
        dec_out = self.phase_detection(x_enc)
        return dec_out  


    def phase_detection(self, x_enc):
        B, M, L = x_enc.shape
        
        # Normalization from Non-stationary Transformer

        #seg_num = 100
        #x_enc = rearrange(x_enc, 'b m l -> b l m')
        #x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        #means = x_enc.mean(2, keepdim=True).detach()
        #x_enc = x_enc - means
        #stdev = torch.sqrt(
        #    torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        #x_enc /= stdev
        #x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        x_enc = self.convs(x_enc)

#        x_enc = self.res_convs(x_enc)

#        x_enc = self.bilstms(x_enc)

        x_enc = rearrange(x_enc, 'b m l -> b l m')

        enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))
        #print(enc_out.shape)
        
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
#         # Process through BERT
#         outputs = self.bert(inputs_embeds=enc_out).last_hidden_state
    
        outputs = outputs[:, :, :self.d_ff]
        # outputs = self.ln_proj(outputs)
        
        dec_out = self.out_layer(outputs)
        
        #print(dec_out.shape)

        # De-Normalization from Non-stationary Transformer

        #dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        #dec_out = dec_out * \
        #          (stdev[:, :, 0, :].unsqueeze(2).repeat(
        #              1, 1, seg_num, 1))
        #dec_out = dec_out + \
        #          (means[:, :, 0, :].unsqueeze(2).repeat(
        #              1, 1, seg_num, 1))
        #dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')
        dec_out = rearrange(dec_out, 'b l m -> b m l')
        
        dec_out = [decoder(dec_out) for decoder in self.decoders]
        
        dec_out = torch.cat(dec_out, dim=1)


        return dec_out 
    
@register_model
def GPT4EQ(**kwargs):
    model = Model(**kwargs)
    return model