import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import math

import lightly
from lightly.loss import NTXentLoss
from utils.utils import get_optimizer, linear_increase_alpha, margin_decay, mse_alignment_loss, get_latest_checkpoint
from utils.transform import positive_transform, negative_transform, PositiveMaskingTransform

from utils.losses import PatchContrastiveLoss 

#from utils.losses import DINOLoss, IBOTPatchLoss
from .neg_sampling import NegSamplerClasses, NegSamplerRandomly, NegSamplerNN, NegSamplerStatic
import timm
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from lightly.loss import KoLeoLoss
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule
import random

class Trainer:
    def __init__(self, model, train_loader, val_loader, args):

        # load model
        self.model = model.to(args.device)
        # load dataloader
        self.train_loader = train_loader
        # training setting
        self.epochs = args.epochs
        self.device = args.device
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.device_id = args.device_id
        self.args = args
        self.save_path = args.save_path
        self.start_epoch = 0
        # choosing mode
        self.mode = args.mode
        self.momentum_ema = args.ema


        ##########################################
        #    Setting loss function for each mode #
        ##########################################
        if self.mode == 'mae':
            self.criterion = nn.MSELoss()
        elif self.mode == 'simclr':
            self.criterion = NTXentLoss(temperature=args.loss_temp)
        elif self.mode == 'simclr_supcon':
            self.criterion = SupConLoss()
        elif self.mode == "dinov2":
            #self.criterion = DINOLoss(output_dim=2048,warmup_teacher_temp_epochs=5,)
            self.criterion1 = DINOLoss()
            self.criterion2 = IBOTPatchLoss()
            self.criterion3 = KoLeoLoss()
            self.criterion = "Total loss DINO, IBOTPatchLoss, KoLeoLoss"
        elif self.mode == "simMIM":
            self.criterion = nn.L1Loss()
        elif self.mode == "SHAM":
            self.criterion1 = NTXentLoss(temperature=args.temp)
            self.criterion2 = PatchContrastiveLoss(temperature=args.temp)
            self.criterion3 = nn.MSELoss()
        
        # optimizer configuration
        self.optimizer = get_optimizer(self.model, self.lr, self.weight_decay, self.beta1, self.beta2)

        # choosing backbone
        self.mode_model = args.model

        ####################################
        #        Loading checkpoint        #
        ####################################
        if self.args.continue_training:
            try:
                latest_ckpt_path = get_latest_checkpoint(args.checkpoint_folder)
                checkpoint = torch.load(latest_ckpt_path, map_location=self.device)
                self.save_path = args.checkpoint_folder
                print(f"✅ Found checkpoint: {latest_ckpt_path}")
            except (FileNotFoundError, TypeError):
                print("⚠️ No valid checkpoint found, starting from scratch.")
                self.start_epoch = 0
                global_loss, local_loss = 0.0, 0.0
            else:
                # Load model weights
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Khởi tạo optimizer mới khớp với model hiện tại
                self.optimizer = get_optimizer(
                    self.model,
                    self.args.lr,
                    self.args.weight_decay,
                    self.args.beta1,
                    self.args.beta2
                )

                # Load optimizer state
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load epoch và các thông tin bổ sung
                self.start_epoch = checkpoint.get('epoch', 0)
                global_loss = checkpoint.get('global_loss', 0.0)
                local_loss = checkpoint.get('local_loss', 0.0)

                print(f"🔁 Loaded checkpoint from epoch {self.start_epoch}")

                # Đảm bảo optimizer trên đúng device (đặc biệt khi map_location='cpu')
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
        else:
            self.start_epoch = 0
            global_loss, local_loss = 0.0, 0.0


        ####################################
        #    Creating saving directory     #
        ####################################    
        self.momentum_ema = args.ema
        if args.mode=="SHAM":
            self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_{self.args.SHAM_mode}")    
        else: 
            self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}")
        if not os.path.exists(self.save_path):
            print(f"Save {args.mode} at {self.save_path}")      
            os.makedirs(self.save_path, exist_ok=True)
            new_log=True
        else:
            new_log=False


        mode = 'a' if not new_log else 'w'
        self.log_file = os.path.join(self.save_path, 'training_log.txt')
        with open(self.log_file, mode) as f: 
            if new_log:
                f.write("Training Log - Loss per Epoch\n")
            else:
                f.write("\n---- Resume training ----\n")

        self.log_dir = os.path.join(self.save_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir) 
    
    def train_one_epoch_simclr(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                images = batch[0]
                x0, x1 = images[0], images[1]
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                z0 = self.model(x0)
                z1 = self.model(x1)

                loss = self.criterion(z0, z1)
                running_loss += loss.detach()

            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad() 
        return running_loss / len(self.train_loader)
    
    def train_one_epoch_mae(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                views = batch[0]
                images = views[0].to(self.device)
                predictions, targets = self.model(images)
                loss = self.criterion(predictions, targets)
                running_loss += loss.detach()
            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        return running_loss / len(self.train_loader)

    def train_one_epoch_simclr_supcon(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training with simclr on supcon"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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

            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        return running_loss / len(self.train_loader)
    
    def train_one_epoch_dinov2(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        self.total_steps = self.epochs * len(self.train_loader)
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training with dinov2")):
            views = batch[0]
            views = [view.to(self.device) for view in views]
            global_views = torch.cat(views[:2])
            local_views = torch.cat(views[2:])

            # Masking
            B = len(global_views)
            sequence_length = self.model.teacher_backbone.sequence_length
            mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)

            H, W = self.model.teacher_backbone.vit.patch_embed.grid_size
            assert (
                H * W == sequence_length - 1
            ), f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"

            block_mask = random_block_mask(size=(B, H, W), device=mask.device)
            mask[:, 1:] = block_mask.flatten(start_dim=1)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Teacher forward
                with torch.no_grad():
                    teacher_cls_token, teacher_features = self.model.forward_teacher(global_views)
                    teacher_cls_out = self.model.teacher_head.dino_head.forward(teacher_cls_token)
                    teacher_masked_out = self.model.teacher_head.ibot_head.forward(
                        teacher_features[mask]
                    )

                # Student forward
                student_global_cls_token, student_global_masked_features = \
                    self.model.forward_student(global_views, mask=mask)

                student_global_cls_out = self.model.student_head.dino_head.forward(
                    student_global_cls_token
                )
                student_global_masked_out = self.model.student_head.ibot_head.forward(
                    student_global_masked_features
                )
                student_local_cls_token, _ = self.model.forward_student(local_views, mask=None)
                student_local_cls_out = self.model.student_head.dino_head.forward(
                    student_local_cls_token
                )
                student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

                # Loss
                global_step = epoch * len(self.train_loader) + batch_idx
                teacher_temp = linear_warmup_schedule(
                    step=global_step,
                    warmup_steps=int(30 / self.epochs * self.total_steps),
                    start_value=0.04,
                    end_value=0.07,
                )
                dino_loss = self.criterion1(
                    teacher_out=teacher_cls_out.chunk(2),
                    student_out=student_cls_out.chunk(len(views)),
                    teacher_temp=teacher_temp,
                )
                ibot_loss = self.criterion2(
                    teacher_out=teacher_masked_out,
                    student_out=student_global_masked_out,
                    mask=block_mask,
                    teacher_temp=teacher_temp,
                )
                koleo_loss = 0.1 * sum(
                    self.criterion3(t) for t in student_global_cls_token.chunk(2)
                )
                loss = dino_loss + ibot_loss + koleo_loss

            running_loss += loss.detach()

            # ✅ Mixed Precision update
            scaler.scale(loss).backward()

            # zero lr for last layer if needed
            if epoch < 1:
                for param_group in self.optimizer.param_groups:
                    if "last_layer" in param_group:
                        param_group["lr"] = 0.0

            # weight decay schedule
            weight_decay = cosine_schedule(
                step=global_step,
                max_steps=self.total_steps,
                start_value=0.04,
                end_value=0.4,
            )
            for group in self.optimizer.param_groups:
                if group["weight_decay"] != 0.0:
                    group["weight_decay"] = weight_decay

            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

            # momentum update teacher
            momentum = cosine_schedule(
                step=global_step,
                max_steps=self.total_steps,
                start_value=0.992,
                end_value=1.0,
            )
            update_momentum(self.model.student_backbone, self.model.teacher_backbone, m=momentum)
            update_momentum(self.model.student_head, self.model.teacher_head, m=momentum)

        avg_loss = running_loss / len(self.train_loader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        return avg_loss

    def train_one_epoch_simMIM(self, epoch=0, alpha=0, scaler=None):
        running_loss =0.0
        for batch in tqdm(self.train_loader, desc="Training with simMIM"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                views = batch[0]
                images = views[0].to(self.device)  # views contains only a single view
                predictions, targets = self.model(images)

                loss = self.criterion(predictions, targets)
                running_loss += loss.detach()

            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
        
        return running_loss/len(self.train_loader)
    
    def train_one_epoch_SHAM(self, epoch=0, momentum_val=0.99, scaler=None):
        self.model.train()
        running_loss_total = 0.0
        running_loss_global = 0.0
        running_loss_local = 0.0
        running_loss_semantic_alignment =0.0

        for batch_id, batch in enumerate(tqdm(self.train_loader, desc="Training with negative samples")):
            self.optimizer.zero_grad()
            images = batch[0]
            current_m = momentum_val 
        
            update_momentum(self.model.backbone, self.model.teacher_backbone, m=current_m)
            update_momentum(self.model.proj_global, self.model.teacher_proj_global, m=current_m)
            update_momentum(self.model.proj_local, self.model.teacher_proj_local, m=current_m)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                x_weak = images['weak'].to(self.device) # teacher input
                x_strong = images['strong'].to(self.device) # student input
                
                res = self.model(img_student=x_strong, img_teacher=x_weak)
                global_s, global_t, local_s, local_t  = res['global_s'], res['global_t'], res['local_s'], res['local_t']
                masked_patches_t, masked_patches_s = res["masked_patches_t"], res["masked_patches_s"]
                
                global_loss = self.criterion1(global_s, global_t)
                local_loss = self.criterion2(local_s, local_t)
                semantic_alignment_loss = self.criterion3(masked_patches_t, masked_patches_s)
                total_loss = global_loss + 0.5*local_loss + 0.2 * semantic_alignment_loss
            
            running_loss_total += global_loss.item()
            running_loss_global += local_loss.item()
            running_loss_local += total_loss.item()
            running_loss_semantic_alignment += semantic_alignment_loss .item()
            
            #total_loss.backward()
            #self.optimizer.step()
            scaler.scale(total_loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(self.optimizer)
            scaler.update()
        
        self.writer.add_scalar('Epoch/Current', epoch, global_step=epoch)
        self.writer.add_scalars(
            'Loss/Avg_per_Epoch',  # Nhóm dưới tag này để dễ xem theo epoch
            {
                'total_loss': running_loss_total/len(self.train_loader),
                'global_loss': running_loss_global/len(self.train_loader),
                'local_loss': running_loss_local/len(self.train_loader)
            },
            global_step=epoch  # Sử dụng epoch trực tiếp làm step cho log epoch-level
        )

        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}: Total Loss = {running_loss_total/len(self.train_loader):.6f}, Global Loss = {running_loss_global/len(self.train_loader):.6f}, Local Loss = {running_loss_local/len(self.train_loader):.6f}, Semantic_alignment Loss: {running_loss_semantic_alignment/len(self.train_loader):.6f}\n")

        return running_loss_total/len(self.train_loader), running_loss_global/len(self.train_loader), running_loss_local/len(self.train_loader), running_loss_semantic_alignment/len(self.train_loader)
    
    def train(self):
        if self.mode == "mae":
            train_one_epoch = self.train_one_epoch_mae
        elif self.mode == 'simclr':
            train_one_epoch = self.train_one_epoch_simclr
        elif self.mode == 'simclr_supcon':
            train_one_epoch = self.train_one_epoch_simclr_supcon
        elif self.mode == "dinov2":
            train_one_epoch = self.train_one_epoch_dinov2
        elif self.mode == "simMIM":
            train_one_epoch = self.train_one_epoch_simMIM
        elif self.mode == "SHAM":
            train_one_epoch = self.train_one_epoch_SHAM

        scaler = torch.cuda.amp.GradScaler() 

        for epoch in range(self.start_epoch, self.epochs):
            print(f"Epoch {epoch}/{self.epochs}")
            if self.mode=="SHAM":
                total_loss, global_loss, local_loss, semantic_alignment_loss = train_one_epoch(epoch=epoch, momentum_val=self.momentum_ema, scaler=scaler)
                print(f"Total train loss: {total_loss:.6f}, Global loss: {global_loss:.6f}, Local loss: {local_loss:.6f}, Semantic_alignment_loss: {semantic_alignment_loss}")
            else:
                train_loss = train_one_epoch(epoch=epoch, alpha=0, scaler=scaler)
                print(f"Train loss: {train_loss:.4f}")
            if (epoch+1) % 20 == 0:
                file_name = os.path.join(self.save_path, f"model_ckpt_{epoch}.pth")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'args': self.args,
                    "total_loss": total_loss,
                    'global_loss': global_loss,
                    'local_loss': local_loss,
                    'semantic_alignment_loss': semantic_alignment_loss
                }
                torch.save(checkpoint, file_name)
                print(f"✅ Saved checkpoint at epoch {epoch} -> {file_name}")
            file_name = os.path.join(self.save_path, f"model_ckpt_latest.pth")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'args': self.args,
                "total_loss": total_loss,
                'global_loss': global_loss,
                'local_loss': local_loss,
                'semantic_alignment_loss': semantic_alignment_loss
            }
            torch.save(checkpoint, file_name)
            print(f"✅ Saved checkpoint at epoch {epoch} -> {file_name}")

        self.writer.close()  # Giải phóng resource
            

