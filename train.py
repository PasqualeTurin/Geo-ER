import torch
from torch import nn, optim
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import pickle

def train_GeoER(model, train_x, train_coord, train_n, train_y, valid_x, valid_coord, valid_n, valid_y, test_x, test_coord, test_n, test_y, device, save_path, epochs=10, batch_size=32, lr=3e-5):

  opt = optim.Adam(params=model.parameters(), lr=lr)
  criterion = nn.NLLLoss()
  
  valid_x_tensor = torch.tensor(valid_x)
  valid_coord_tensor = torch.tensor(valid_coord)
  valid_y_tensor = torch.tensor(valid_y)

  test_x_tensor = torch.tensor(test_x)
  test_coord_tensor = torch.tensor(test_coord)
  test_y_tensor = torch.tensor(test_y)

  num_steps = (len(train_x) // batch_size) * epochs
  scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

  best_f1 = 0.0

  for epoch in range(epochs):
    model.train()
    print('\n*** EPOCH:',epoch+1,'***\n')

    i = 0
    step = 1

    while i < len(train_x):
      
      opt.zero_grad()
      loss = torch.tensor(0.).to(device)

      if i + batch_size > len(train_x):
        y = train_y[i:]
        x = train_x[i:]
        x_coord = train_coord[i:]
        x_n = train_n[i:]
      else:
        y = train_y[i : i + batch_size]
        x = train_x[i : i + batch_size]
        x_coord = train_coord[i : i + batch_size]
        x_n = train_n[i : i + batch_size]



      y = torch.tensor(y).view(-1).to(device)
      x = torch.tensor(x)
      x_coord = torch.tensor(x_coord)
      att_mask = torch.tensor(np.where(x != 0, 1, 0))

      pred = model(x, x_coord, x_n, att_mask)

      loss = criterion(pred, y)

      loss.backward()
      opt.step()

      if device == 'cuda':
        print('Step:',step,'Loss:',loss.cpu().detach().numpy())
      else:
        print('Step:',step,'Loss:',loss)

      step += 1
      scheduler.step()
      i += batch_size


    print('\n*** Validation Epoch:',epoch+1,'***\n')
    this_f1 = validate_GeoER(model, valid_x_tensor, valid_coord_tensor, valid_n, valid_y_tensor, device)

    print('\n*** Test Epoch:',epoch+1,'***\n')
    _ = validate_GeoER(model, test_x_tensor, test_coord_tensor, test_n, test_y_tensor, device)


    if this_f1 > best_f1:
      best_f1 = this_f1
      pickle.dump(model, open(save_path, 'wb'))


def validate_GeoER(model, valid_x_tensor, valid_coord_tensor, valid_n, valid_y_tensor, device):

  attention_mask = np.where(valid_x_tensor != 0, 1, 0)
  attention_mask = torch.tensor(attention_mask)
  model.eval()

  acc = 0.0
  prec = 0.0
  recall = 0.0
  f1 = 0.0

  for i in range(valid_x_tensor.shape[0]):

    y = valid_y_tensor[i].view(-1).to(device)
    x = valid_x_tensor[i]
    x_coord = valid_coord_tensor[i]
    x_n = valid_n[i:i+1]
    att_mask = attention_mask[i]

    pred = model(x, x_coord, x_n, att_mask, training=False)
    
    if torch.argmax(pred) == y:
      acc += 1

      if y == 1:
        recall += 1

    if torch.argmax(pred) == 1:
      prec += 1

  acc = acc/valid_x_tensor.shape[0]
  if prec > 0:
    prec = recall/prec
  recall = recall/torch.sum(valid_y_tensor).numpy()

  if prec + recall > 0:
    f1 = 2*prec*recall/(prec+recall)
  print("Accuracy:",acc)
  print("Precision:",prec)
  print("Recall:",recall)
  print("f1-Score:",f1)
  
  return f1
