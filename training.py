import cv2
import numpy as np
import torch
import pickle
import torch.nn as nn
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import struct
import PIL
import torchvision.models as models
from data import DATA_DICT
from model_eval import trainer

def get_model(model_name, mode, experiment):
  if mode == "baseline_train":
    if model_name == "vgg":
      model = models.vgg19_bn(pretrained=True)
      model.classifier[-1] = nn.Linear(in_features=4096, out_features=200)
    elif model_name == "googlenet":
      model = models.googlenet(pretrained=True)
      model.fc.out_features = 200
    elif model_name == "resnet":
      model = models.resnet18(pretrained=True)
      model.avgpool = nn.AdaptiveAvgPool2d(1)
      model.fc.out_features = 200
    return model
  else:
    model = torch.load("models/" + model_name + "/" + experiment + ".model")
    return model

def train_experiment(model, model_name, experiment, gpu_boole):
  optimizer = torch.optim.Adam(model.parameters(), 0.001)
  loss_metric = torch.nn.CrossEntropyLoss()
  if gpu_boole:
    model = model.cuda()
  data = DATA_DICT[experiment]
  train_runner = trainer(model, data, optimizer, loss_metric, 25, gpu_boole, 1, 1, 1, 0)
  train_acc, epoch_loss, _, validation_acc,_ = train_runner.train_model()

  results_pickle = {'train_acc': train_acc,
                       'epoch_loss': epoch_loss,
                       'val_acc': validation_acc}

  pickle.dump(results_pickle, open('train_data/' + model_name + '/' + experiment +  '.pkl', 'wb'))
  torch.save(model.state_dict(), 'models/' + model_name + '/' + experiment + '.model')

def eval_experiment(model, model_name, experiment, gpu_boole):
  optimizer = torch.optim.Adam(model.parameters(), 0.001)
  loss_metric = torch.nn.CrossEntropyLoss()
  if gpu_boole:
    model = model.cuda()
  data = DATA_DICT[experiment]
  train_runner = trainer(model, data, optimizer, loss_metric, 25, gpu_boole, 1, 1, 1, 0)
  validation_acc = train_runner.validation_eval()
  results_pickle = {'val_acc': validation_acc}
  pickle.dump(results_pickle, open('results/' + model_name + '/' + experiment +  '.pkl', 'wb'))

def training(model_name, experiment_list, gpu_boole):
  for experiment in experiment_list:
    if experiment == "baseline":
      mode = "baseline_train"
    else:
      mode = "augmented_train"
    model = get_model(model_name, mode, experiment)
    train_experiment(model, model_name, experiment, gpu_boole)

def eval(model_name, experiment_list, gpu_boole):
  mode = "eval"
  for experiment in experiment_list:
    model = get_model(model_name, mode, experiment)
    eval_experiment(model, model_name, experiment, gpu_boole)

if __name__ == "__main__":
  gpu_boole = torch.cuda.is_available()
  model_list = ["vgg", "googlenet", "resnet"]
  evaluation_experiment_list = ["Evaluation_Blur", "Evaluation_Rotation", "Evaluation_Jitter", "Evaluation_Grayscale", "Evaluation_Erasing"]
  training_experiment_list = ["baseline", "Blur", "Rotation", "Jitter", "Grayscale", "Erasing"]
