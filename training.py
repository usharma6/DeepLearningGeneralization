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

def get_model(model_name, mode):
  if model_name == "vgg":
    model = models.vgg19_bn(pretrained=True)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=200)
    if mode == "augmented":
      return model
    elif mode == "baseline":
      for param in model.parameters():
        param.requires_grad = False
      for param in model.classifier[-1].parameters():
        param.requires_grad = True
      return model
  elif model_name == "googlenet":
    model = models.googlenet(pretrained=True)
    model.fc.out_features = 200
    if mode == "augmented":
      return model
    elif mode == "baseline":
      for param in model.parameters():
        param.requires_grad = False
      for param in model.fc.parameters():
        param.requires_grad = True
      return model
  elif model_name == "resnet":
    model = models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc.out_features = 200
    if mode == "augmented":
      return model
    elif mode == "baseline":
      for param in model.parameters():
        param.requires_grad = False
      for param in model.fc.parameters():
        requires_grad = True
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

  pickle.dump(results_pickle, open('results/' + model_name + '/' + experiment +  '.pkl', 'wb'))
  torch.save(model.state_dict(), 'models/' + model_name + '/' + experiment + '.model')

def training(model_name, experiment_list, mode, gpu_boole):
  for experiment in experiment_list:
    model = get_model(model_name, mode)
    train_experiment(model, model_name, experiment, gpu_boole)

if __name__ == "__main__":
  gpu_boole = torch.cuda.is_available()
  model_list = ["vgg", "googlenet", "resnet"]
  baseline_experiment_list = ["Baseline_Blur", "Baseline_Rotation", "Baseline_Jitter", "Baseline_Grayscale", "Baseline_Erasing"]
  augmented_experiment_list = ["Blur", "Rotation", "Jitter", "Grayscale", "Erasing"]
  experiment_list = [baseline_experiment_list, augmented_experiment_list]
  mode_list = ["baseline", "augmented"]
  for model in model_list:
      for i in range(len(mode_list)):
          training(model, experiment_list[i], mode_list[i], gpu_boole)
