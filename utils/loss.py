# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np



class Energy_Square_Hinge_Loss(nn.Module):

    def __init__(self, energy_inlier_margin = -15, energy_outlier_margin = -5):
        super().__init__()
        self.energy_inlier_margin = energy_inlier_margin
        self.energy_outlier_margin = energy_outlier_margin
        
    def forward(self, energy_score_inlier, energy_socre_outlier):

        inlier_hinge = torch.mean(torch.square(torch.relu(energy_score_inlier - self.energy_inlier_margin)))
        outlier_hinge = torch.mean(torch.square(torch.relu(self.energy_outlier_margin - energy_socre_outlier)))
        return inlier_hinge + outlier_hinge




class Reconstruction_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, label, model_output):

        loss = torch.mean(
                    torch.sum(
                        torch.square(label - model_output) , dim = (1,2,3)
                    )
                )

        return loss

class Similarity_Triplet_Loss(nn.Module):

    def __init__(self, margin = 10):
        super().__init__()
        self.margin = margin
        

    def forward(self, anchor_features, positive_features, negative_features):

        positive_distance = torch.sum(
                                torch.square(anchor_features - positive_features), dim = (1,2,3)
                            )
    
        nagative_distance = torch.sum(
                                torch.square(anchor_features - negative_features), dim = (1,2,3)
                            ) 
    
        # triple loss
        loss = torch.mean(
                   torch.relu(positive_distance + self.margin - nagative_distance)
                )

        return loss