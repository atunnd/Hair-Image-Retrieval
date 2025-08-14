import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

import lightly
from lightly.loss import NTXentLoss
from utils.losses import SupConLoss
from utils.utils import get_optimizer, linear_increase_alpha
from utils.transform import positive_transform 

from .backbone import DINO
from .neg_sampling import NegSamplerClasses, NegSamplerRandomly, NegSamplerMiniBatch
import timm
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum

class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = args.epochs
        self.device = args.device
        self.save_path = args.save_path
        self.mode = args.mode
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.device_id = args.device_id

        if self.mode == 'mae':
            self.criterion = nn.MSELoss()
        elif self.mode == 'simclr':
            self.criterion = NTXentLoss()
        elif self.mode == 'simclr_supcon':
            self.criterion = SupConLoss()
        
        self.optimizer = get_optimizer(self.model, self.lr, self.weight_decay, self.beta1, self.beta2)
        self.neg_sampling = False
        self.neg_loss = args.neg_loss
        self.warm_up_epochs = self.epochs

        if args.neg_sample:
            # neg samling
            self.neg_sampling = True
            self.negative_centroid = args.negative_centroid
            self.warm_up_epochs = args.warm_up_epochs
            self.centroid_momentum = args.centroid_momentum
            self.sampling_frequency = args.sampling_frequency

            print("Training with centroid: ", self.negative_centroid)

            if self.negative_centroid:
                self.save_path = os.path.join(self.save_path, f"{self.mode}_neg_sample_centroid")
                print("Training with neg sample centroid")
            else:
                self.save_path = os.path.join(self.save_path, f"{self.mode}_neg_sample")
                print("Training with neg sample")
            os.makedirs(self.save_path, exist_ok=True)
            print("Create save directory: ", self.save_path)

            # set sampling method and loss
            if self.neg_loss == "simclr":
                self.neg_sampler = NegSamplerMiniBatch(k=5, dim=128, momentum=self.centroid_momentum, device=self.device, device_id=self.device_id, negative_centroid=self.negative_centroid, save_path=self.save_path)
                self.criterion = NTXentLoss()
                print("create kmeans with faiss")
            elif self.neg_loss == "supcon":
                self.neg_sampler = NegSamplerClasses()
                self.criterion = SupConLoss()
            
            # init triplet loss
            self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        else:
            self.save_path = os.path.join(self.save_path, self.mode)
            os.makedirs(self.save_path, exist_ok=True)
    
    def train_one_epoch_simclr(self, epoch=0, alpha=0):
        self.model.train()
        running_loss = 0.0
        for batch, _ in tqdm(self.train_loader, desc="Training"):
            images = batch[0]
            x0, x1 = images[0], images[1]
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            z0 = self.model(x0)
            z1 = self.model(x1)
            loss = self.criterion(z0, z1)
            running_loss += loss.detach()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad() 
        return running_loss / len(self.train_loader)
    
    def train_one_epoch_mae(self, epoch=0):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            views = batch[0]
            images = views[0].to(self.device)
            predictions, targets = self.model(images)
            loss = self.criterion(predictions, targets)
            running_loss += loss.detach()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return running_loss / len(self.train_loader)

    def train_one_epoch_simclr_supcon(self, epoch=0, alpha=0):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training with simclr on supcon"):
            images, labels = batch[0], batch[1]
            images = [img.to(self.device) for img in images]
            labels = labels.to(self.device)
            images = torch.cat([images[0], images[1]], dim=0)
            bsz = labels.shape[0]
            
            features = self.model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = self.criterion(features, labels)
            running_loss += loss.detach()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return running_loss / len(self.train_loader)

    def train_one_epoch_simclr_neg_supervised(self, epoch=0, alpha=0):
        self.model.train()
        running_loss =0.0
        for batch in tqdm(self.train_loader, desc="Training with supervised negative sampling"):
            images, labels = batch[0], batch[1].to(self.device)
            images = [img.to(self.device) for img in images]

    
    def train_one_epoch_simclr_neg_sample(self, epoch=0, alpha=0, neg_batch=None, momentum_val=0):
        self.model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        for batch_id, batch in enumerate(tqdm(self.train_loader, desc="Training with negative samples")):
            images, _ = batch[0], batch[1].to(self.device)
            update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum_val)
            update_momentum(
                self.model.projection_head, self.model.projection_head_momentum, m=momentum_val
            )
            images = [img.to(self.device) for img in images]
            trip_loss = 0.0

            ### STAGE 1: Randomly negative sampling
            if self.warm_up_epochs > epoch + 1:
                negative_samples = positive_transform(NegSamplerRandomly(images[0]))
                neg_batch[batch_id] = self.model.forward_momentum(negative_samples)

            ### STAGE 2: Hard negative mining
            else:
                if (epoch + 1) % self.sampling_frequency == 0:
                    if batch_id == 0:
                        print("Init centroids")
                    ema_embeddings = self.model.forward_momentum(images[0])
                    neg_batch[batch_id] = self.neg_sampler.forward(ema_embeddings, batch_id)

            pos_samples = positive_transform(images[1])
            pos_batch = self.model.forward_momentum(pos_samples)
            anchor_batch = self.model.forward_momentum(images[0])
            #print(anchor_batch.shape, pos_batch.shape, neg_batch[batch_id].shape)
            trip_loss = self.triplet_loss(anchor_batch, pos_batch, neg_batch[batch_id])
            running_loss1 += trip_loss.item()
            
            # Main encoder running
            x0, x1 = images
            x0, x1 = x0.to(self.device), x1.to(self.device)
            z0 = self.model(x0)
            z1 = self.model(x1)
            nt_xent_loss = self.criterion(z0, z1)
            running_loss2 += nt_xent_loss.item()
       
            # Total loss
            total_loss = nt_xent_loss + alpha*trip_loss
            running_loss += total_loss.detach()

            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return running_loss/len(self.train_loader),running_loss1/len(self.train_loader), running_loss2/len(self.train_loader), neg_batch

    
    def train(self):
        if self.mode == "mae":
            train_one_epoch = self.train_one_epoch_mae
        elif self.mode == 'simclr':
            train_one_epoch = self.train_one_epoch_simclr
        elif self.mode == 'simclr_supcon':
            train_one_epoch = self.train_one_epoch_simclr_supcon

        print(f"Training model {self.mode} with losses {self.criterion}")
        
        if self.neg_sampling:
            train_one_epoch = self.train_one_epoch_simclr_neg_sample
            neg_batch= [torch.Tensor([]) for _ in range(len(self.train_loader))]
            

        for epoch in range(self.epochs):
            alpha = 1
            if self.neg_sampling:
                if self.warm_up_epochs <= epoch + 1:
                    alpha = linear_increase_alpha(start_alpha=.001, current_epoch=(epoch+1)-self.warm_up_epochs, max_epochs=self.epochs-self.warm_up_epochs)
                else:
                    alpha = 0.001
            print(f"Epoch {epoch}/{self.epochs}")
            if self.neg_sampling:
                momentum_val = cosine_schedule(epoch, self.epochs, 0.996, 1)
                train_loss, train_trip_loss, train_ntxent_loss, neg_batch = train_one_epoch(epoch=epoch, alpha=alpha, neg_batch=neg_batch, momentum_val=momentum_val)
                print(f"Total train loss: {train_loss:.4f}, Triplet loss: {train_trip_loss}, NT-Xent loss: {train_ntxent_loss},  Alpha: {alpha}")
            else:
                train_loss = train_one_epoch(epoch=epoch, alpha=alpha)
                print(f"Train loss: {train_loss:.4f}")
            if (epoch+1) % 20 == 0:
                output = os.path.join(self.save_path, f"model_ckpt_{epoch}.pth")
                torch.save(self.model.state_dict(), output)
                print(f"✅ Model saved to {self.save_path}")
            

