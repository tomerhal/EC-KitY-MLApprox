import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset

from sklearn.model_selection import KFold

import numpy as np

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_encodings.ga.vector_individual import Vector

from net import Net
from utils import *

class NeuralNetworkEvaluator(SimpleIndividualEvaluator):
    def __init__(self, trainset, batch_size, n_epochs):
        super().__init__()
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainset = trainset

    def _evaluate_individual(self, individual: Vector) -> float:
        # Create a neural network with the individual's parameters
        model = Net(*individual.vector)

        # Initialize optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        train_samples = np.random.choice(range(CIFAR10_TRAIN_SAMPLES), size=int(CIFAR10_TRAIN_SAMPLES * 0.8), replace=False)
        val_samples = [i for i in range(CIFAR10_TRAIN_SAMPLES) if i not in train_samples]

        train_set = Subset(self.trainset, train_samples)
        val_set = Subset(self.trainset, val_samples)

        # Create a DataLoader for the training data
        trainloader = DataLoader(
            train_set, batch_size=self.batch_size,
            num_workers=1, pin_memory=True)

        # Create a DataLoader for the validation data
        valloader = DataLoader(
            val_set, batch_size=self.batch_size,
            num_workers=1, pin_memory=True)
        
        model.train()

        epochs_accuracies = dict()

        for epoch in range(1, self.n_epochs+1):
            self.train_epoch(model, trainloader, criterion, optimizer)
            accuracy = self.val_epoch(model, valloader)
            epochs_accuracies[epoch] = accuracy

        weights = list(epochs_accuracies.keys())
        return np.average(list(epochs_accuracies.values()), weights=weights)
    

    def kfold_cv(self, model, criterion, optimizer):
        kf = KFold(n_splits=5, shuffle=True)
        accuracies = []

        for train_idx, val_idx in kf.split(self.trainset):
            # Create a DataLoader for the training data for this fold
            trainloader = DataLoader(
                self.trainset, batch_size=self.batch_size, sampler=SubsetRandomSampler(train_idx),
                num_workers=2, pin_memory=True)

            # Create a DataLoader for the validation data for this fold
            valloader = DataLoader(
                self.trainset, batch_size=self.batch_size, sampler=SubsetRandomSampler(val_idx),
                num_workers=2, pin_memory=True)
            
            epochs_accuracies = dict()
            for epoch in range(1, self.n_epochs+1):
                self.train_epoch(model, trainloader, criterion, optimizer)
                accuracy = self.val_epoch(model, valloader)
                epochs_accuracies[epoch] = accuracy

            fold_accuracy = np.average(list(epochs_accuracies.values()), weights=list(epochs_accuracies.keys()))
            accuracies.append(fold_accuracy)

        return np.mean(accuracies)
            

    def train_epoch(self, model, trainloader, criterion, optimizer) -> None:
        for images, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            images,labels = images.to(self.device),labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()


    def val_epoch(self, model, valloader):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                # Move data and target to GPU if available
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                output = model(images)

                # Compute the predicted class
                _, predicted = torch.max(output.data, 1)

                # Compute the accuracy
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
