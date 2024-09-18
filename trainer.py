# -*- coding: utf-8 -*-

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import numpy as np
import cv2      # for save images
import imageio  # for making GIF
from PIL import Image
import matplotlib.pyplot as plt

from utils.dataset import downstream_task_dataset, collate_fn
from utils.model import Downstream_Task_Model
from utils.loss import Energy_Square_Hinge_Loss

import os
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
from datetime import datetime
import shutil


class model_trainer():

    def __init__(self,args):

        self.mode = args.mode.lower()
        self.gpu = args.gpu

        self.load_ckpt = args.load_ckpt
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.optim = args.optimizer
        self.no_log = args.no_log
        self.note = args.note

        self.lamda = args.lamda
        self.ood_threshold = args.ood_threshold
        self.energy_inlier_margin = args.energy_inlier_margin
        self.energy_outlier_margin = args.energy_outlier_margin

        
        self.time_slot = datetime.today().strftime("%Y%m%d_%H%M")
        self.logdir = os.path.join(os.path.dirname(__file__), self.time_slot + "_logs" + self.note)
        self.ckpt_dir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok = True)
                    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[!] torch version: {torch.__version__}")
        print(f"[!] computation device: {self.device}, index : {self.gpu}")
        print(f"[!] execution mode: {self.mode}")
    

    def __printer(info):
        def wrap1(function):
            def wrap2(self , *args, **argv):
                print(f"[!] {info}...")
                function(self , *args, **argv)
                print(f"[!] {info} Done.")
            return wrap2
        return wrap1


    @__printer("Data Loading")
    def load_data(self):

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        transforms_train = torchvision.transforms.Compose( [
                                    # torchvision.transforms.Lambda(lambda x: 2. * (np.array(x) / 255.) - 1.),
                                    # torchvision.transforms.Lambda(lambda x: torch.from_numpy(x).float()),
                                    # torchvision.transforms.Lambda(lambda x: torch.permute(x, (2,0,1))),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Pad(4, padding_mode="reflect"),
                                    torchvision.transforms.RandomCrop(32),
                                    torchvision.transforms.RandomHorizontalFlip(p = 0.5),
                                    # torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
                                    torchvision.transforms.Normalize(mean, std),
                                    torchvision.transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))
                                ])
        
        transforms_test = torchvision.transforms.Compose( [
                                    torchvision.transforms.ToTensor(),                            
                                    # torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
                                    torchvision.transforms.Normalize(mean, std),
                                ])

        

        self.train_dataset_in = downstream_task_dataset(train_stage=True, transform=transforms_train, OOD=False)
        self.train_dataset_out = downstream_task_dataset(train_stage=True, transform=transforms_train, OOD=True)

        self.test_dataset_in = downstream_task_dataset(train_stage=False, transform=transforms_test, OOD=False)
        self.test_dataset_out = downstream_task_dataset(train_stage=False, transform=transforms_test, OOD=True)

        
        self.train_loader_in = DataLoader(dataset = self.train_dataset_in, batch_size = self.batch_size, 
                                            shuffle = True, num_workers = 1, collate_fn = collate_fn)
        
        self.train_loader_out = DataLoader(dataset = self.train_dataset_out, batch_size = self.batch_size, 
                                            shuffle = True, num_workers = 1, collate_fn = collate_fn)

        self.test_loader_in = DataLoader(dataset = self.test_dataset_in, batch_size = self.batch_size,
                                        shuffle = False, num_workers = 1, collate_fn = collate_fn)
        
        self.test_loader_out = DataLoader(dataset = self.test_dataset_out, batch_size = self.batch_size,
                                        shuffle = False, num_workers = 1, collate_fn = collate_fn)
                                    
                                        
    @__printer("Setup")
    def setup(self):
        
        # define our model, loss function, and optimizer
        self.log_writer = None
        if self.no_log is False:
            self.log_writer = SummaryWriter(self.logdir)
            self.record_args(self.logdir)
        

        self.EnergyModel = Downstream_Task_Model().to(self.device)
        self.show_parameter_size(self.EnergyModel, "EnergyModel")

        self.CrossEntropy = torch.nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.Energy_Square_Hinge_Loss = Energy_Square_Hinge_Loss(self.energy_inlier_margin, self.energy_outlier_margin).to(self.device)
        

        trainable_model_parameters = []
        trainable_model_parameters += self.EnergyModel.parameters()

        if self.optim.lower() == "adam":
            self.optimizer = torch.optim.Adam(trainable_model_parameters, lr=self.lr)
        elif self.optim.lower() == "rmsprop":
            self.optimizer = torch.optim.RMSprop(trainable_model_parameters, lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(trainable_model_parameters, lr=self.lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
        
        
        def warmup_cosine_annealing(step, total_steps, lr_max, lr_min):
            warm_up_iter = 1000
            if step < warm_up_iter:
                return step / warm_up_iter
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                        lr_lambda=lambda step: warmup_cosine_annealing(step, self.epochs * len(self.train_loader_in),
                                                                1,  1e-6 / self.lr))# since lr_lambda computes multiplicative factor

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = "min" , factor = 0.5, patience = 20,  min_lr = 1e-6)

        # load checkpoint file to resume training
        if self.load_ckpt:
            self.load()



    def execute(self):

        if self.mode == "baseline_train":
            self.train(energy_train_flag=False)
            self.save("baseline_train_end")

        elif self.mode == "energy_train":
            self.train(energy_train_flag=True)
            self.save("energy_train_end")

    

    @__printer("Model Training")
    def train(self, energy_train_flag = True):
        print("\n")

        avg_time = 0
    
        # evaluation metrics
        self.best_loss = float("inf")
        self.best_acc = -1e8
        self.best_FPR = -1e8
        self.best_AUROC = -1e8
        self.state = {"train_loss":[], "test_loss":[], "test_acc":[], "test_FPR95":[], "test_AUROC":[]}

        for epoch in range(self.epochs):

            st = perf_counter()
            
            self.EnergyModel.train()
            train_loss = 0
            for i , (inlier_batch, outlier_batch) in tqdm(enumerate(zip(self.train_loader_in, self.train_loader_out)), desc="Train Progress", leave=False):

                in_data, in_targets = inlier_batch[0].to(self.device), inlier_batch[1].to(self.device)
                out_data, _ = outlier_batch[0].to(self.device), outlier_batch[1].to(self.device)

                inlier_logit, inlier_energy = self.EnergyModel(in_data)
                outlier_logit, outlier_energy = self.EnergyModel(out_data)

                loss = self.CrossEntropy(inlier_logit, in_targets)
                if energy_train_flag:
                    loss_energy = self.Energy_Square_Hinge_Loss(inlier_energy, outlier_energy)
                    loss += self.lamda * loss_energy


                # updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # evaluation
                # train_loss += (loss.item() / n_train_total_steps)
                train_loss = 0.9 * train_loss + 0.1 * loss.item()

            self.state["train_loss"].append(train_loss)

            test_num = 0
            test_loss = 0
            test_accuracy = 0
            test_OOD_FPR_at_TPR95 = 0
            test_inlier_score_list = []
            test_outlier_score_list = []
            self.EnergyModel.eval()
            with torch.no_grad():
                for i , (inlier_batch, outlier_batch) in tqdm(enumerate(zip(self.test_loader_in, self.test_loader_out)), desc="Test Progress", leave=False):
        
                    in_data, in_targets = inlier_batch[0].to(self.device), inlier_batch[1].to(self.device)
                    out_data, _ = outlier_batch[0].to(self.device), outlier_batch[1].to(self.device)

                    inlier_logit, inlier_energy = self.EnergyModel(in_data)
                    outlier_logit, outlier_energy = self.EnergyModel(out_data)


                    loss = self.CrossEntropy(inlier_logit, in_targets)
                    if energy_train_flag:
                        loss_energy = self.Energy_Square_Hinge_Loss(inlier_energy, outlier_energy)
                        loss += self.lamda * loss_energy

                    # test_loss += (loss.item() / n_test_total_steps)
                    test_loss = 0.9 * test_loss + 0.1 * loss.item()


                    inlier_prediction = self.EnergyModel.posterior_predict(logit = inlier_logit)
                    outlier_prediction = self.EnergyModel.posterior_predict(logit = outlier_logit)

                    test_num += in_data.shape[0]
                    test_accuracy += torch.sum(torch.argmax(inlier_prediction, dim = -1) == in_targets)

                    if energy_train_flag:
                        test_inlier_score_list.append(-1 * inlier_energy.detach().cpu().numpy())
                        test_outlier_score_list.append(-1 * outlier_energy.detach().cpu().numpy())

                    else:
                        in_softmax_score = torch.max(inlier_prediction, dim = -1)[0].detach().cpu().numpy()
                        out_softmax_score = torch.max(outlier_prediction, dim = -1)[0].detach().cpu().numpy()
                        test_inlier_score_list.append(in_softmax_score)
                        test_outlier_score_list.append(out_softmax_score)


            test_accuracy = (test_accuracy / test_num).detach().cpu().numpy()
            test_OOD_FPR_at_TPR95 = self.calculate_FPR_at_TPR95(test_inlier_score_list, test_outlier_score_list)
            test_AUROC, test_TPRs, test_FPRs = self.calculate_AUROC(test_inlier_score_list, test_outlier_score_list)

            self.state["test_loss"].append(test_loss)
            self.state["test_acc"].append(test_accuracy)
            self.state["test_FPR95"].append(test_OOD_FPR_at_TPR95)
            self.state["test_AUROC"].append(test_AUROC)

            self.log_writer.add_scalar(f"Train Loss" , train_loss , epoch)
            self.log_writer.add_scalar(f"Test Loss" , test_loss , epoch)
            self.log_writer.add_scalar(f"Test Acc." , test_accuracy , epoch)
            self.log_writer.add_scalar(f"Test OOD FPR@TPF95" , test_OOD_FPR_at_TPR95 , epoch)
            self.log_writer.add_scalar(f"Test AUROC" , test_AUROC , epoch)
        

            avg_time = avg_time + (perf_counter() - st - avg_time) / (epoch+1)
            print(f"[!] ┌── Epoch: [{epoch+1}/{self.epochs}] done, Training time per epoch: {avg_time:.3f}")
            print(f"[!] ├── Train Loss: {train_loss:.4f}")
            print(f"[!] ├── Test Loss: {test_loss:.4f}")
            print(f"[!] ├── Acc.: {test_accuracy:.4f}, OOD FPR@TPR95: {test_OOD_FPR_at_TPR95:.4f}, AUROC: {test_AUROC:.4f}")
            print(f"[!] └──────────────────────────────────────────────────────────────\n")

            if test_accuracy >= self.best_acc:
                self.best_epoch = epoch
                self.best_loss = test_loss
                self.best_acc = test_accuracy
                self.best_FPR = test_OOD_FPR_at_TPR95
                self.best_AUROC = test_AUROC
                # self.save(f"best_{epoch:04d}")
                self.save(f"best")

                plt.plot(np.array(test_FPRs), np.array(test_TPRs), color = "r")
                plt.title(f"ROC, area under ROC = {test_AUROC:.4f}, FPR@TPR95 = {test_OOD_FPR_at_TPR95:.4f}")
                plt.xlabel("FPR"); plt.ylabel("TPR"); plt.xlim(0,1); plt.ylim(0,1)
                plt.savefig(os.path.join(self.logdir, "test_ROC.png")); plt.close()
                

            if epoch % 10 == 0:
                print(f"[!] Best Epoch: {self.best_epoch}, Loss: {self.best_loss:.4f}, Acc.: {self.best_acc:.4f}, FPR@TPR95: {self.best_FPR:.4f}, AUROC: {self.best_AUROC:.4f}\n") 

        if not self.no_log:
            self.log_writer.close()

        for k, v in self.state.items():
            plt.plot(np.arange(len(v)), np.array(v), color = "r")
            plt.title(k); plt.xlabel("epoch")
            plt.savefig(os.path.join(self.logdir, f"{k}.png")); plt.close()

        print(f"[!] Best Epoch: {self.best_epoch}, Loss: {self.best_loss:.4f}, Acc.: {self.best_acc:.4f}, FPR@TPR95: {self.best_FPR:.4f}, AUROC: {self.best_AUROC:.4f}\n") 


    @__printer("Model Saving")    
    def save(self , name = ""):

        keys = ["EnergyModel"]
        values = [self.EnergyModel.state_dict()]

        checkpoint = {
            k:v for k, v in zip(keys, values) 
        }
        
        torch.save(checkpoint , os.path.join(self.ckpt_dir, f"ckpt_{name}.pth"))



    @__printer("Model Loading")  
    def load(self):
        checkpoint = torch.load(self.load_ckpt)
        self.EnergyModel.load_state_dict(checkpoint['EnergyModel'])


    def calculate_FPR_at_TPR95(self, inlier_scores, outlier_scores):

        inlier_scores = np.concatenate(inlier_scores, axis = 0)
        outlier_scores = np.concatenate(outlier_scores, axis = 0)

        index = int(inlier_scores.shape[0] * 0.05)
        TPR95_thres = np.sort(inlier_scores)[index]
        FPR = (outlier_scores >= TPR95_thres).sum() / outlier_scores.shape[0]
        return FPR


    def calculate_AUROC(self, inlier_scores, outlier_scores):

        inlier_scores = np.sort(np.concatenate(inlier_scores, axis = 0))
        outlier_scores = np.sort(np.concatenate(outlier_scores, axis = 0))
        
        TPRs, FPRs = [], []
        P, N = inlier_scores.shape[0], outlier_scores.shape[0]
        prev_th = None

        th_list =np.concatenate([np.array([1e8]), inlier_scores[::-1], np.array([-1e8])], axis = 0)
        for th in th_list:
            if th == prev_th:
                continue
            TP = (inlier_scores >= th).sum()
            FP = (outlier_scores >= th).sum()
            TPRs.append(TP / P)
            FPRs.append(FP / N)
            prev_th = th

        auroc = 0
        for i in range(len(FPRs)-1):
            fpr0, fpr1 = FPRs[i], FPRs[i+1]
            tpr0, tpr1 = TPRs[i], TPRs[i+1]
            w = np.abs(fpr1 - fpr0)
            h = (tpr1 + tpr0) / 2
            auroc += w*h

        return auroc, TPRs, FPRs

    def train_logging(self , epoch , n_total_steps , i):

        if not self.no_log:
            with torch.no_grad():

                self.Encoder.eval()
                self.Decoder.eval()
                for TYPE , DATA in [("Train", self.train_pair_for_log) , ("Test", self.test_pair_for_log)]:
                
                    inputs, labels = DATA[0][100:120].to(self.device) , DATA[1][100:120].to(self.device)
                    features = self.Encoder(inputs)
                    outputs = self.Decoder(features)
                    # input_grid = torchvision.utils.make_grid(inputs , nrow = 4 , normalize = True)
                    output_grid = torchvision.utils.make_grid(outputs , nrow = 4 , normalize = True)
                    label_grid = torchvision.utils.make_grid(labels , nrow = 4 , normalize = True)

                    big_grid = torch.cat([label_grid , torch.zeros([3 , label_grid.size(1) , 10]).to(self.device), output_grid] , axis = 2)
            
                    torchvision.utils.save_image(big_grid , os.path.join("." , f"{self.time_slot}_logs@{self.note}@" ,  f"{self.time_slot}_{TYPE}_GIF" , f"{epoch * n_total_steps + i + 1}-iteration.png"))
                    self.log_writer.add_image(f"{TYPE} Pair: label_grid, output_grid" , big_grid , epoch * n_total_steps + i + 1)
    
    def make_gif(self):

        non_sorted_list = list(Path(self.logdir).rglob("*.png"))
        if len(non_sorted_list) == 0:
            print(f"[!] No such images in folder to make GIF. Please check")
            return

        png_list = sorted(non_sorted_list, key = lambda x: int(str(x).split(os.sep)[-1].split("-")[0]))

        process = [cv2.imread(str(i) , 0) for i in png_list]
        imageio.mimsave(os.path.join(self.logdir , f"processing.gif") , process , fps = 10)
        # [os.remove(i) for i in png_list]
        

    
    def record_args(self , path):

        source_code_path = os.path.dirname(__file__)
        backup_path = os.path.join(path, "src_backup")
        os.makedirs(backup_path, exist_ok=True)
        shutil.copy(os.path.join(source_code_path, "trainer.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "main.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "dataset.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "loss.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "model.py"), backup_path)

        with open(os.path.join(path , "command_args.txt") , "w") as file:
            file.write(f"mode = {self.mode}\n")
            file.write(f"gpu = {self.gpu}\n")
            file.write(f"load ckpt = {self.load_ckpt}\n")
            file.write(f"learning rate = {self.lr}\n")
            file.write(f"optimizer = {self.optim}\n")
            file.write(f"batch size = {self.batch_size}\n")
            file.write(f"epochs = {self.epochs}\n")
            file.write(f"lambda = {self.lamda}\n")
            file.write(f"ood threshold = {self.ood_threshold}\n")
            file.write(f"energy inlier margin = {self.energy_inlier_margin}\n")
            file.write(f"energy outlier margin = {self.energy_outlier_margin}\n")
        return


    def show_parameter_size(self, model, model_name = "Model"):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"[!] {model_name} - number of parameters: {params}")
