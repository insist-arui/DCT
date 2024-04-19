#c类型  res2 cat res3  res4 cat res5  y_low cat y_high
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch as tr
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from mmcv.cnn import ConvModule
from mmengine.model.weight_init import constant_init, normal_init, xavier_init, trunc_normal_, kaiming_init
from typing import Optional, Tuple, Union
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType, SampleList
from .vit_utils import DropPath
from typing import Optional, Tuple, Union

class PatchEmebed(nn.Module):
	def __init__(self,dim = 768) -> None:
		super().__init__()
		self.dim = dim
		self.conv1 = nn.Conv2d(256,dim,kernel_size=7,stride=8,padding=0)
		self.conv2 = nn.Conv2d(512,dim,kernel_size=4,stride=4,padding=0)
		self.conv3 = nn.Conv2d(1024,dim,kernel_size=2 ,stride=2,padding=0)
		self.conv4 = nn.Conv2d(2048,dim,kernel_size=1,stride=1,padding=0)#bt,768,7,7
		self.embed = nn.ModuleList([self.conv1,self.conv2,self.conv3,self.conv4])
		#self.proj1 = nn.Linear(98,98)
		#self.proj2 = nn.Linear(98,98)
		self.init_weights()
	def init_weights(self):
		"""Initialize weights."""
		# Lecun norm from ClassyVision
		kaiming_init(self.conv1, mode='fan_in', nonlinearity='linear')
		kaiming_init(self.conv2, mode='fan_in', nonlinearity='linear')
		kaiming_init(self.conv3, mode='fan_in', nonlinearity='linear')
		kaiming_init(self.conv4, mode='fan_in', nonlinearity='linear')
	def forward(self, x:Tuple[torch.Tensor]) -> torch.Tensor:
		out = []
		b,c,t,w,h = x[-1].size()               #b c t w h
		for i ,x_back in enumerate(x):
			B,C,T,W,H = x_back.size()
			x_back = x_back.transpose(2,1).reshape(-1,C,W,H)#bt c w h
			out_ = self.embed[i](x_back) #BT 768 7 7
			out_ = rearrange(out_, 'a c w h -> a c (w h)') #bt 768 49
			out.append(out_)
		# y = torch.cat((out[0],out[1],out[2],out[3]),dim=-1)
		y1 = torch.cat((out[0],out[1]),dim=-1) #bt 768 98
		y2 = torch.cat((out[2],out[3]),dim=-1)
		#y1 = self.proj1(y1)
		#y2 = self.proj2(y2)
		y1 = y1.transpose(1,2)
		y2 = y2.transpose(1,2)
		return y1,y2,h,t,b
class CosPairwise(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, q, k):
		q_norm = q / q.norm(dim=-1)[:, :, :, None]
		k_norm = k / k.norm(dim=-1)[:, :, :, None]

		return torch.matmul(q_norm, k_norm.transpose(-2,-1))
class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True, attention_type=None):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5
		self.with_qkv = with_qkv
		if self.with_qkv:
			self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
			self.proj = nn.Linear(dim, dim)
			self.proj_drop = nn.Dropout(proj_drop)
		self.attn_drop = nn.Dropout(attn_drop)

		self.attention_type = attention_type
		if self.attention_type == 'divided_space_time':
			self.cos = CosPairwise()

	def forward(self, x):
		B, N, C = x.shape  #B,196*8,M
		if self.with_qkv:
			qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#3,B,8,196*8,64
			q, k, v = qkv[0], qkv[1], qkv[2]#B,8,196*8,64
		else:
			qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
			q, k, v  = qkv, qkv, qkv#B,8,196*8,64

		if self.attention_type == 'divided_space_time':
			attn = self.cos(q*self.scale, k*self.scale)#B,8,196*8,196*8
		else:
			attn = (q @ k.transpose(-2, -1)) * self.scale
			attn = attn.softmax(dim=-)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		if self.with_qkv:
			x = self.proj(x)
			x = self.proj_drop(x)
		return x




class DoubleAttention(nn.Module):
	def __init__(self,dim,num_heads,attn_drop=0., proj_drop=0.,attention_type=None):
		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5
		self.qkv1 = nn.Linear(dim,dim*3)
		self.qkv2 = nn.Linear(dim,dim*3)
		self.proj = nn.Linear(dim, dim)

		self.attn_drop = nn.Dropout(attn_drop)
		self.proj_drop = nn.Dropout(proj_drop)
		self.cos = CosPairwise()
		self.attention_type = attention_type
		# self.FourierAttention = FourierAttention(dim,dim,seq_len_q=frame, seq_len_kv=frame)
	def forward(self,x1,x2):
		B, N, C = x1.shape

		qkv1 = self.qkv1(x1).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
		qkv2 = self.qkv2(x2).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
		q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
		q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

		if self.attention_type == 'temporal':
			att = self.cos(q1*self.scale, k2*self.scale)
		else:
			att = (q1 @ k2.transpose(-2,-1)) * self.scale

		att = att.softmax(dim=-1)
		att = self.attn_drop(att)
		y = (att @ v2).transpose(1, 2).reshape(B, N, C)
		y = self.proj(y)


		# y = torch.cat((y1,y2),2)
		return y
class DoubleCrossAttention(nn.Module):
	def __init__(self,dim,num_heads,frame,attn_drop=0., proj_drop=0.,attention_type=None):
		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5
		self.qkv = nn.Linear(dim,dim*3)
		self.proj1 = nn.Linear(dim, dim)
		self.proj2 = nn.Linear(dim, dim)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj_drop = nn.Dropout(proj_drop)
		self.cos = CosPairwise()
		self.attention_type = attention_type
		self.FourierAttention = FourierAttention(dim,dim,seq_len_q=frame, seq_len_kv=frame)
	def forward(self,x1,x2):
		B, N, C = x1.shape
		qkv1 = self.qkv(x1).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
		qkv2 = self.qkv(x2).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
		q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
		q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
		if self.attention_type == 'temporal':
			att1 = self.cos(q1*self.scale, k2*self.scale)
			att2 = self.cos(q2*self.scale, k1*self.scale)
		else:
			att1 = (q1 @ k2.transpose(-2,-1)) * self.scale
			att2 = (q2 @ k1.transpose(-2,-1)) * self.scale

		att1 = att1.softmax(dim=-1)
		att1 = self.attn_drop(att1)
		y1 = (att1 @ v2).transpose(1, 2).reshape(B, N, C)
		y1 = self.proj1(y1)
		att2 = att2.softmax(dim=-1)
		att2 = self.attn_drop(att2)
		y2 = (att2 @ v1).transpose(1,2).reshape(B, N, C)
		y2 = self.proj2(y2)

		# y = torch.cat((y1,y2),2)
		return y1,y2

class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x

class Block(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4.,drop=0., attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
		super().__init__()
		self.attention_type = attention_type
		assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])
		self.norm1 = norm_layer(dim)
		self.spatial_attn1 = Attention(dim,num_heads,attn_drop)
		self.spatial_attn2 = Attention(dim, num_heads, attn_drop)
		## Temporal Attention Parameters
		if self.attention_type == 'divided_space_time':
			self.temporal_norm1 = norm_layer(dim)
			self.temporal_norm2 = norm_layer(dim)
			self.temporal_norm = norm_layer(dim)
			# self.temporal_attn = Attention(
			# dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attention_type=self.attention_type)
			self.temporal_attn1 = DoubleAttention(dim,num_heads,attn_drop,attention_type='temporal')
			self.temporal_attn2 = DoubleAttention(dim, num_heads, attn_drop, attention_type='temporal')
			self.temporal_fc1 = nn.Linear(dim, dim)
			self.temporal_fc2 = nn.Linear(dim, dim)
			self.temporal_fc = nn.Linear(dim, dim)
		## drop path
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		self.norm = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
		self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
		self.tattn = Attention(dim,num_heads,attn_drop,attention_type='temporal')
		self.sattn = Attention(dim, num_heads, attn_drop)
	def forward(self, x1,x2, B, T, W):
		num_spatial_tokens = (x1.size(1) - 1) // T  #x b,98*8+1,m   w=7
		H = num_spatial_tokens // W   #H=14
		if self.attention_type == 'divided_space_time':
			#self low dimension temporal
			# xt2 = x2[:,1:,:]#B,98*8,M
			xt11 = x1[:,1:,:]#B,98*8,M
			xt11 = rearrange(xt11, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
			xt11 = self.tattn(self.temporal_norm(xt11))
			res_temporal = self.drop_path(xt11)
			res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
			res_temporal = self.temporal_fc(res_temporal)
			xt = x1[:,1:,:] + res_temporal

			#self low dimension spatial
			init_cls_token = x1[:, 0, :].unsqueeze(1)
			cls_token = init_cls_token.repeat(1, T, 1)
			cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
			xs = xt
			xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
			xs = torch.cat((cls_token, xs),1)
			res_spatial = self.drop_path(self.sattn(self.norm(xs)))
			res_spatial = res_spatial + xs   #(bt)(hw+1)m


			#across attention temporal 1
			xt1 = res_spatial[:,1:,:]  #bt,hw,m
			xt2 = x2[:,1:,:]         #b hwt m
			xt1 = rearrange(xt1,'(b t)(h w) m ->(b h w) t m',b=B,h=H,w=W,t=T)
			xt2 = rearrange(xt2, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
			res_temporal1 = self.temporal_attn1(self.temporal_norm1(xt1), self.temporal_norm2(xt2))
			res_temporal1 = self.drop_path(res_temporal1)
			res_temporal1 = rearrange(res_temporal1, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
			res_temporal1 = self.temporal_fc1(res_temporal1)    #b (h w t) m
			xt1 = rearrange(xt1,'(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
			xt1 = xt1 + res_temporal1       #b hwt m
			#need res_spatial head
			int_cls_token1 = res_spatial[:,0,:] #bt,m
			cls_token1 = int_cls_token1
			cls_token1 = cls_token1.unsqueeze(1)  #bt,1,m
			xs1 = rearrange(xt1,'b (h w t) m ->(b t)(h w) m',b=B, h=H, w=W, t=T)
			xs1 = torch.cat((cls_token1,xs1),1)    #bt hw+1 m
			res_spatial1 = self.drop_path(self.spatial_attn1(self.norm1(xs1)))
			cls_token1 = res_spatial1[:,0,:]
			cls_token1 = rearrange(cls_token1,'(b t) m ->b t m',b=B,t=T)
			cls_token1 = torch.mean(cls_token1, 1, True)
			res_spatial1 = res_spatial1[:,1:,:]
			res_spatial1 = rearrange(res_spatial1, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)

			int_cls_token1 = rearrange(int_cls_token1,'(b t) m -> b t m',b=B, t=T)
			init_cls_token1 = torch.mean(int_cls_token1, 1, True)
			x = torch.cat((init_cls_token1, xt1), 1) + torch.cat((cls_token1, res_spatial1), 1)  # b,h w t +1,m
			x_low = x + self.drop_path(self.mlp1(self.norm1(x))) #b,hwt+1,m
			y1 = x_low
            #high  x2 x_low
			#temporal
			x_high = x2[:,1:,:]
			x_low = x_low[:,1:,:]
			x_low = rearrange(x_low,'b (h w t) m -> (b h w) t m',b=B, h=H, w=W, t=T)
			x_high = rearrange(x_high, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
			res_temporal2 = self.drop_path(self.temporal_attn2(x_high,x_low))
			res_temporal2 = rearrange(res_temporal2, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
			res_temporal2 = self.temporal_fc2(res_temporal2)
			xt_high = x2[:,1:,:] +res_temporal2
			#spatial
			init_cls_token2 = x2[:, 0, :].unsqueeze(1)
			cls_token2 = init_cls_token2.repeat(1, T, 1)
			cls_token2 = rearrange(cls_token2, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
			xs_high = xt_high
			xs_high = rearrange(xs_high, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
			xs_high = torch.cat((cls_token2, xs_high), 1)
			res_spatial_high = self.drop_path(self.spatial_attn2(self.norm2(xs_high)))  # bt hw+1 m
			cls_token2 = res_spatial_high[:, 0, :]
			cls_token2 = rearrange(cls_token2, '(b t) m -> b t m', b=B, t=T)
			cls_token2 = torch.mean(cls_token2, 1, True)  ## averaging for every frame
			res_spatial_high = res_spatial_high[:, 1:, :]
			res_spatial_high = rearrange(res_spatial_high, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)

			x = torch.cat((init_cls_token2, xt_high), 1) + torch.cat((cls_token2, res_spatial_high), 1)  # b,h w t +1,m
			y2 = x + self.drop_path(self.mlp2(self.norm2(x)))
			return y1,y2


@MODELS.register_module()
class DoubleTransformer(nn.Module):
	def __init__(self,frame:int,depth:int,embed_dim=768,num_patches=98,num_heads=8,mlp_ratio=4.,attn_drop_rate=0.,drop_rate=0.,attention_type='divided_space_time',norm_layer=nn.LayerNorm,drop_path_rate=0.1):
		super().__init__()
		self.patch_embed = PatchEmebed()
		self.attention_type = attention_type
		self.depth = depth
		self.head_embedding = nn.Linear(2*embed_dim,embed_dim)
		self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
		self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
		self.time_embed1 = nn.Parameter(torch.zeros(1, frame, embed_dim))
		self.time_embed2 = nn.Parameter(torch.zeros(1, frame, embed_dim))
		self.time_drop = nn.Dropout(p=drop_rate)
		trunc_normal_(self.pos_embed1, std=.02)
		trunc_normal_(self.pos_embed2, std=.02)
		trunc_normal_(self.cls_token1, std=.02)
		trunc_normal_(self.cls_token2, std=.02)
		trunc_normal_(self.time_embed1, std=.02)
		trunc_normal_(self.time_embed2, std=.02)
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
		self.blocks = nn.ModuleList([
			Block(
				dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio,
				drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
			for i in range(self.depth)])
	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
	def forward(self, x):
		x1,x2,H,T,b = self.patch_embed(x)#bt,98,768
		
		B = x1.size(0)
		cls_token1 = self.cls_token1.expand(B, -1, -1)
		cls_token2 = self.cls_token2.expand(B, -1, -1)
		x1 = torch.cat((cls_token1, x1), dim=1)
		x2 = torch.cat((cls_token2, x2), dim=1)
		#print(x1.size(),'111')
		x1 = x1 + self.pos_embed1 #bt,99,768
		x2 = x2 + self.pos_embed2
		if self.attention_type != 'space_only':
			cls_token1 = x1[:b, 0, :].unsqueeze(1)
			cls_token2 = x2[:b, 0, :].unsqueeze(1)
			x1 = x1[:, 1:]#bt,98,768
			x1 = rearrange(x1, '(b t) n m -> (b n) t m',b=b,t=T)#8*98,8,768
			## Resizing time embeddings in case they don't match
			x1 = x1 + self.time_embed1 #1，8，768       8*196，8，768
			x1 = self.time_drop(x1)
			x1 = rearrange(x1, '(b n) t m -> b (n t) m',b=b,t=T)
			x1 = torch.cat((cls_token1, x1), dim=1) #b,nt+1,m

			x2 = x2[:, 1:]  # bt,98,768
			x2 = rearrange(x2, '(b t) n m -> (b n) t m', b=b, t=T)  # 8*98,8,768
			## Resizing time embeddings in case they don't match
			x2 = x2 + self.time_embed2  # 1，8，768       8*98，8，768
			x2 = self.time_drop(x2)
			x2 = rearrange(x2, '(b n) t m -> b (n t) m', b=b, t=T)
			x2 = torch.cat((cls_token2, x2), dim=1)


		for blk in self.blocks:
			x1,x2 = blk (x1, x2,b, T, H)
		# x1_class = x1[:,0,:]
		x1_class = x1[:,0,:] #4,768
		x2_class = x2[:,0,:]
		x = torch.cat((x1_class,x2_class),dim=-1)
		x = self.head_embedding(x)
		return x