import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import struct
import PIL
import torchvision.models as models

class trainer:
    def __init__(self, netparam, dataloaders, optimizer, loss_metric, epochs, gpu_boole, verbose, train_bool, val_bool, test_bool):
        self.netparam = netparam
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.epochs = epochs
        self.gpu_boole = gpu_boole
        self.verbose = verbose
        self.train_bool = train_bool
        self.val_bool = val_bool
        self.test_bool = test_bool

    def train_eval(self):
        correct = 0
        total = 0
        loss_sum = 0
        for images, labels in self.dataloaders['train']:
            if self.gpu_boole:
                images, labels = images.cuda(), labels.cuda()
            outputs = self.netparam(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.float() == labels.float()).sum()
            loss_sum += self.loss_metric(outputs,labels).detach()

        if self.verbose:
            print('Train accuracy: %f %%' % (100.0 * correct / total))
            print('Train loss: %f' % (loss_sum.cpu().data.numpy().item() / total))

        return 100.0 * correct / total, loss_sum.cpu().data.numpy().item() / total


    def test_eval(self):
        correct = 0
        total = 0
        loss_sum = 0
        for images, labels in self.dataloaders['test']:
            if self.gpu_boole:
                images, labels = images.cuda(), labels.cuda()
            outputs = self.netparam(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.float() == labels.float()).sum()
            loss_sum += self.loss_metric(outputs,labels).detach()

        if self.verbose:
            print('Test accuracy: %f %%' % (100.0 * correct / total))
            print('Test loss: %f' % (loss_sum.cpu().data.numpy().item() / total))

        return 100.0 * correct / total, loss_sum.cpu().data.numpy().item() / total

    def validation_eval(self):
        correct = 0
        total = 0
        loss_sum = 0
        for images, labels in self.dataloaders['val']:
            if self.gpu_boole:
                images, labels = images.cuda(), labels.cuda()
            outputs = self.netparam(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.float() == labels.float()).sum()
            loss_sum += self.loss_metric(outputs,labels).detach()

        if self.verbose:
            print('Validation accuracy: %f %%' % (100.0 * correct / total))
            print('Validation loss: %f' % (loss_sum.cpu().data.numpy().item() / total))

        return 100.0 * correct / total, loss_sum.cpu().data.numpy().item() / total



    def train_model(self):
        loss_batch_store = []
        train_accuracy_scores = []
        test_accuracy_scores = []
        validation_accuracy_scores = []
        training_loss = []
        epoch_train_loss = []
        print("Starting Training")

        #training loop:
        for epoch in range(self.epochs):
          time1 = time.time() #timekeeping
          for i, (x, y) in enumerate(self.dataloaders['train']):
            if self.gpu_boole:
              x = x.cuda()
              y = y.cuda()

            outputs = self.netparam.forward(x)
            loss = self.loss_metric(outputs,y)
            loss.backward()

            loss_batch_store.append(loss.cpu().data.numpy().item())
            self.optimizer.step()
            self.optimizer.zero_grad()

          print("Epoch",epoch+1,':')
          if self.train_bool:
              train_perc, train_loss = self.train_eval()
              train_accuracy_scores.append(train_perc)
              epoch_train_loss.append(train_loss)
          if self.val_bool:
              valid_perc, valid_loss = self.validation_eval()
              validation_accuracy_scores.append(valid_perc)
          if self.test_bool:
              test_perc, test_loss = self.test_eval()
              test_accuracy_scores.append(test_perc)


          time2 = time.time() #timekeeping
          print('Elapsed time for epoch:',time2 - time1,'s')
          print('ETA of completion:',(time2 - time1)*(self.epochs - epoch - 1)/60,'minutes')
          print()

        return train_accuracy_scores, epoch_train_loss, loss_batch_store, validation_accuracy_scores, test_accuracy_scores
