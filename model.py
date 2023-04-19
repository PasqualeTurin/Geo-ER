import config
from transformers import BertModel, BertTokenizer
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

class GeoER(nn.Module):
  def __init__(self, device='cpu', finetuning=True, c_emb=config.c_em, n_emb=config.n_em, a_emb=config.a_em, dropout=0.1):
      super().__init__()

      hidden_size = config.lm_hidden

      self.language_model = BertModel.from_pretrained('bert-base-uncased')
      self.neighbert = BertModel.from_pretrained('bert-base-uncased')
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

      self.device = device
      self.finetuning = finetuning

      self.drop = nn.Dropout(dropout)
      self.attn = nn.Linear(hidden_size, 1)
      self.linear1 = nn.Linear(hidden_size + 2*c_emb + n_emb, (hidden_size + 2*c_emb + n_emb)//2)
      self.linear2 = nn.Linear((hidden_size + 2*c_emb + n_emb)//2, 2)

      self.neigh_linear = nn.Linear(2*a_emb, n_emb)
      self.coord_linear = nn.Linear(1, 2*c_emb)

      self.attn = nn.Linear(2*a_emb, 1)
      self.w_attn = nn.Linear(hidden_size, a_emb)
      self.b_attn = nn.Linear(1,1)

      self.relu = nn.ReLU()
      self.gelu = nn.GELU()
      self.tanh = nn.Tanh()
      self.leaky = nn.LeakyReLU()



  def forward(self, x, x_coord, x_n, att_mask, training=True):


    x = x.to(self.device)
    att_mask = att_mask.to(self.device)
    x_coord = x_coord.to(self.device)
    self.neighbert.eval()

    if len(x.shape) < 2:
      x = x.unsqueeze(0)

    if len(att_mask.shape) < 2:
      att_mask = att_mask.unsqueeze(0)

    while len(x_coord.shape) < 2:
      x_coord = x_coord.unsqueeze(0)

    b_s = x.shape[0]

    if training and self.finetuning:
      self.language_model.train()
      self.train()
      output = self.language_model(x, attention_mask=att_mask)
      pooled_output = output[0][:, 0, :] # take only 0 (the position of the [CLS])

    else:

      self.language_model.eval()
      with torch.no_grad():
        output = self.language_model(x, attention_mask=att_mask)
        pooled_output = output[0][:, 0, :]
        
    
    x_neighbors = []
    for b in range(b_s):
    
      x_neighborhood1 = []
      x_neighborhood2 = []
      with torch.no_grad():
        x_node1 = torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n[b]['name1'])['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], 1).squeeze()
        x_node2 = torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n[b]['name2'])['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], 1).squeeze()
      
        for x_n1 in x_n[b]['neigh1']:
          x_neighborhood1.append(torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n1)['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], 1).squeeze())
        
        if not len(x_neighborhood1):
          x_neighborhood1.append(torch.zeros(768))
          
        for x_n2 in x_n[b]['neigh2']:
          x_neighborhood2.append(torch.mean(self.neighbert(torch.tensor(self.tokenizer(x_n2)['input_ids']).to(self.device).unsqueeze(0))[0][:, :, :], 1).squeeze())
        
        if not len(x_neighborhood2):
          x_neighborhood2.append(torch.zeros(768))
          
        x_neighborhood1 = torch.stack(x_neighborhood1).to(self.device)
        x_neighborhood2 = torch.stack(x_neighborhood2).to(self.device)
        
        x_distances1 = x_n[b]['dist1']
        if not len(x_distances1):
          x_distances1.append(1000)
          
        x_distances2 = x_n[b]['dist2']
        if not len(x_distances2):
          x_distances2.append(1000)
          
        x_distances1 = torch.tensor(x_distances1, dtype=torch.float).view(-1, 1).to(self.device)
        x_distances2 = torch.tensor(x_distances2, dtype=torch.float).view(-1, 1).to(self.device)

      x_concat1 = torch.cat([self.w_attn(x_node1).view(1,-1).repeat(x_neighborhood1.shape[0], 1), self.w_attn(x_neighborhood1)], 1)
      x_concat2 = torch.cat([self.w_attn(x_node2).view(1,-1).repeat(x_neighborhood2.shape[0], 1), self.w_attn(x_neighborhood2)], 1)

      x_att1 = F.softmax(self.leaky(self.attn(x_concat1)) + self.b_attn(x_distances1),0)
      x_att2 = F.softmax(self.leaky(self.attn(x_concat2)) + self.b_attn(x_distances2),0)

      x_context1 = torch.sum(self.w_attn(x_neighborhood1)*x_att1,0)
      x_context2 = torch.sum(self.w_attn(x_neighborhood2)*x_att2,0)

      x_sim1 = x_context1*self.w_attn(x_node1)
      x_sim2 = x_context2*self.w_attn(x_node2)

      x_neighbors.append(self.relu(torch.cat([x_sim1, x_sim2])))

    
    x_neighbors = torch.stack(x_neighbors)
    x_neighbors = self.neigh_linear(x_neighbors)
    
    x_coord = x_coord.transpose(0,1)
    x_coord = self.coord_linear(x_coord)

    output = torch.cat([pooled_output, x_coord, x_neighbors], 1)

    output = self.linear2(self.drop(self.gelu(self.linear1(output))))
    
    return F.log_softmax(output, dim=1)
