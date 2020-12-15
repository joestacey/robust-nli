
#Importing packages

import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet
from models import BiasNet
from models import LSTMEncoder

parser = argparse.ArgumentParser(description='NLI training')

# paths
parser.add_argument("--nlipath", type=str, default='/home/jstacey/GABi/dataset/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="/home/jstacey/GABi/dataset/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--min_epoch", type=int, default=5, help="min number of epochs the model uses the full learning rate for")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# Initialising multiple bias classifiers arguments
parser.add_argument("--start_bias_classifiers", type=int, default=20, help="Number of bias classifiers at the start")
parser.add_argument("--start_adv_classifiers", type=int, default=1, help="Number of adversary classifiers at the start")


# Remote adjustment parameters
parser.add_argument("--update_embeddings", type=float, default=1, help="Update embeddings or not")
parser.add_argument("--adv_loss_multiplier", type=float, default=0.5, help="Multiplier for the adversarial models in the loss term")
parser.add_argument("--load_model", type=float, default=0, help="Load pre-trained model or not")
parser.add_argument("--load_id", type=str, default="", help="model ID to load")
parser.add_argument("--save_id", type=str, default="", help="model ID to be saved")
parser.add_argument("--load_classifier", type=float, default=0, help="Load pre-trained classifier or not")
parser.add_argument("--update_classifier", type=float, default=1, help="Update classifier or not")


# Data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# Set gpu device
torch.cuda.set_device(params.gpu_id)

# Initialising multiple bias models and adversaries

adv_models = range(1, params.start_adv_classifiers + 1)
bias_models = range(1, params.start_bias_classifiers + 1)

# Print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])


"""
MODEL
"""
# Model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'use_cuda'       :  True                  ,

}

# Model encoders available (currently set to LSTMEncoder)
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']


nli_net = NLINet(config_nli_model)
print(nli_net)
nli_net.cuda()

encoder_net = LSTMEncoder(config_nli_model)
print(encoder_net)
encoder_net.cuda()

bias_list = []

# Loss

# NLI model loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn_main = nn.CrossEntropyLoss(weight=weight)
loss_fn_main.size_average = False
loss_fn_main.cuda()


def create_bias_loss(classes):
    weight = torch.FloatTensor(params.n_classes).fill_(1)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    loss_fn.size_average = False

    return loss_fn


bias_loss_functions = []

for i in bias_models:
    bias_loss_functions.append(create_bias_loss(params.n_classes))
    bias_list.append(BiasNet(config_nli_model))

for i in bias_list:
    i.cuda()

for i in bias_loss_functions:
    i.cuda()

print(bias_loss_functions)

# Optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# optimizer list for the bias models

bias_optimizers = []

for i in bias_list:
    bias_optimizers.append(optim_fn(i.parameters(), **optim_params))

encoder_net.cuda()

encoder_optimizer = optim_fn(encoder_net.parameters(), **optim_params)

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()

    correct_main = 0.

    correct_bias_list = []
    for i, j in enumerate(bias_models):
        correct_bias_list.append(0.)

    # Shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    batch_number = 0

    for stidx in range(0, len(s1), params.batch_size):

        batch_number = batch_number + 1
               
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch = Variable(s1_batch.cuda())
        s2_batch = Variable(s2_batch.cuda())
       
        # Input
        
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size

        # NLI model forward

        output_main = nli_net(encoder_net((s1_batch, s1_len)), encoder_net((s2_batch, s2_len)))        

        pred = output_main.data.max(1)[1]
        correct_main += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # Bias models forward
        output_bias_list = []
        
        for i, j in enumerate(bias_loss_functions):
            output_bias_list.append(bias_list[i](encoder_net((s1_batch, s1_len)),encoder_net((s2_batch, s2_len))))
            pred = output_bias_list[i].data.max(1)[1]
            correct_bias_list[i] += pred.long().eq(tgt_batch.data.long()).cpu().sum()
          
        # Loss main NLI task

        loss_main = loss_fn_main(output_main, tgt_batch)
        
        # Loss for bias models

        loss_list = []
        
        for i, j in enumerate(bias_loss_functions):
            loss_list.append(bias_loss_functions[i](output_bias_list[i], tgt_batch))

        # Loss for encoder

        loss_encoder = (1-params.adv_loss_multiplier/(len(adv_models)))*loss_main

        for adv_model_no in adv_models:
            loss_encoder = loss_encoder - params.adv_loss_multiplier*loss_list[adv_model_no-1]/len(adv_models)

        # Backwards for main model
                
        optimizer.zero_grad()
        loss_main.backward(retain_graph=True)

        # Gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                if p.grad is not None:
                    p.grad.data.div_(k)
                    total_norm += p.grad.data.norm() ** 2

        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr']
              
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor
        
        # Optimizer step
        if params.update_classifier:

            optimizer.step()
 
        optimizer.param_groups[0]['lr'] = current_lr

        # Backwards for bias models

        for i, j in enumerate(loss_list):
            bias_optimizers[i].zero_grad()
            loss_list[i].backward(retain_graph=True)
            # Gradient clipping
            shrink_factor = 1
            total_norm = 0
            for p in bias_list[i].parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        p.grad.data.div_(k)  # divide by the actual batch size
                        total_norm += p.grad.data.norm() ** 2
            total_norm = np.sqrt(total_norm)

          if total_norm > params.max_norm:
              shrink_factor = params.max_norm / total_norm
          current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)

          bias_optimizers[i].param_groups[0]['lr'] = current_lr * shrink_factor # just for update

          # Optimizer step
          bias_optimizers[i].step()
 
          bias_optimizers[i].param_groups[0]['lr'] = current_lr

        # Backwards for the encoder
        
        encoder_optimizer.zero_grad()
        loss_encoder.backward(retain_graph=True)

        # Gradient clipping
        shrink_factor = 1
        total_norm = 0

        for p in encoder_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)

        encoder_optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
      
        # Optimizer step
        if params.update_embeddings:       
          encoder_optimizer.step()
         
        encoder_optimizer.param_groups[0]['lr'] = current_lr

        if batch_number == 100:

          print("Printing training accuray in Epoch",epoch," so far:")
          print("Main NLI task accuracy:",round(100.*correct_main/(stidx+k), 5))
          print((stidx+k),"observations analysed during epoch")
          
          for i, correct_value in enumerate(correct_bias_list):
            print("Bias model",str(i),"accuracy:",round(100.*correct_value/(stidx+k), 5))              

def evaluate(epoch, adv_models, bias_models, bias_loss_functions, bias_list, bias_optimizers, optim_params, eval_type='valid', final_eval=False):
  nli_net.eval()

  correct_main = 0.
  correct_bias_list = []

  for i in bias_models:
    correct_bias_list.append(0.)
 
  global val_acc_best, lr, stop_training, adam_stop

  if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

  s1 = valid['s1'] if eval_type == 'valid' else test['s1']
  s2 = valid['s2'] if eval_type == 'valid' else test['s2']
  target = valid['label'] if eval_type == 'valid' else test['label']

  for i in range(0, len(s1), params.batch_size):
        # Prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # NLI model forward
        output_main = nli_net(encoder_net((s1_batch, s1_len)), encoder_net((s2_batch, s2_len)))
       
        pred = output_main.data.max(1)[1]
 
      
        correct_main += pred.long().eq(tgt_batch.data.long()).cpu().sum()

        # Bias models forward
        output_bias_list = []

        for t, j in enumerate(correct_bias_list):

          output_bias_list.append(bias_list[t](encoder_net((s1_batch, s1_len)), encoder_net((s2_batch, s2_len))))
          pred = output_bias_list[t].data.max(1)[1]
          correct_bias_list[t] += pred.long().eq(tgt_batch.data.long()).cpu().sum()

  # Printing accuracy
  eval_acc = round(100 * correct_main / len(s1), 5)
  print("Main NLI task accuracy:",eval_acc)

  for t, correct_value in enumerate(correct_bias_list):
    print("Bias model",str(t),"accuracy:",round(100.*correct_value/(len(s1)), 5))

  # Updating learning rate / potentially stopping model
                
  adv_correct_scores = []
  not_adv_correct_scores = []

  correct_bias_list = [100 * x/len(s1) for x in correct_bias_list]

  for t, j in enumerate(correct_bias_list):

    if (t+1) in adv_models:
      adv_correct_scores.append(j)

    else: 
      not_adv_correct_scores.append(j)

  if eval_type == 'valid' and epoch <= params.n_epochs:
    if eval_acc > val_acc_best:
      val_acc_best = eval_acc
    else:
      if 'sgd' in params.optimizer:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
        print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
        if optimizer.param_groups[0]['lr'] < params.minlr:
             stop_training = True
      if 'adam' in params.optimizer:
        stop_training = adam_stop
        adam_stop = True
  return eval_acc, adv_models, bias_models, bias_loss_functions, bias_list, bias_optimizers


"""
Train model on Natural Language Inference task
"""
epoch = 1

if params.load_model:
  encoder_net.load_state_dict(torch.load('../encoder_' + params.load_id + 'saved.pt'))
 
if params.load_classifier:
  nli_net.load_state_dict(torch.load('../classifier_' + params.load_id + 'saved.pt')) 

while (not stop_training or epoch <= params.min_epoch) and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)

    eval_acc, adv_models, bias_models, bias_loss_functions, bias_list, bias_optimizers = evaluate(epoch, adv_models, bias_models, bias_loss_functions, bias_list, bias_optimizers, optim_params, 'valid')
    epoch += 1

print("Evaluating on test-set")

evaluate(0, adv_models, bias_models, bias_loss_functions, bias_list, bias_optimizers, optim_params, 'test', True)


torch.save(encoder_net.state_dict(),'../encoder_' + params.save_id + 'saved.pt')
torch.save(nli_net.state_dict(),'../classifier_' + params.save_id + 'saved.pt')
