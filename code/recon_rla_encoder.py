"""
Point-based scalable point-to-query encoder based on the work of RandLA-Net (https://arxiv.org/pdf/1911.11236.pdf). The detailed explanation can be found in the Bachelor thesis report of Nicolas Muntwyler (municola@student.ethz.ch).
You can find a short summary of the hyperparameters in experiments/recon_rla/configs/config_current.yaml.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ddr.utils import randla_utils as pt_utils
import numpy as np

class ReconRLA_Encoder(nn.Module):
    def __init__(self, num_layers, d_out, latent_size, enable_middle_mlp, num_aggregations, aggregation_neighbors, include_normals):
        super().__init__()
        
        # Debugging Attributs
        debug = False # Set this to True for debugging (Includes KNN visualization files)
        self.debug = debug
        self.debugCounter = 0

        # Network Architecture settings
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.d_out = d_out
        self.enable_middle_mlp = enable_middle_mlp
        self.num_aggregations = num_aggregations
        self.aggregation_neighbors = aggregation_neighbors
        self.include_normals = include_normals
        
        # Inlucde Normals or Not (Recommended: Include them)
        if include_normals:
            self.fc0 = pt_utils.Conv1d(6, 8, kernel_size=1, bn=True)
        else:
            self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)
        
        # Create the Encoding layer LFA modules
        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.num_layers):
            d_out = self.d_out[i]//2
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out
        d_out = d_in
        
        # Enable or Disable the MLP in the middle of the U-Net
        if self.enable_middle_mlp:
            self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

        # Create the Decoding FA modules
        self.decoder_blocks2 = nn.ModuleList()
        for j in range(self.num_layers):
            # Dimension of: layer before, skip connection, next layer
            dim_lb = self.d_out[-j-1]
            if j == self.num_layers - 1:
                dim_sc = self.d_out[-j-1]
                dim_nl = self.latent_size
            else:
                dim_sc = self.d_out[-j-2]
                dim_nl = self.d_out[-j-2]
            
            self.decoder_blocks2.append(Decoder_lfa(debug=self.debug, 
                                                    dim_lb=dim_lb,
                                                    dim_sc=dim_sc,
                                                    dim_nl=dim_nl,
                                                    num_agg=self.num_aggregations,
                                                    agg_k=self.aggregation_neighbors))

    def forward(self,features, points, queries, points_batch, queries_batch, batch_size, inputs):
        if (self.debug):
            import pdb
            pdb.set_trace()
            torch.save(features, '/home/nicolas/eth/ba/ddr/experiments/randla/results/debug/points.pt')
            torch.save(queries, '/home/nicolas/eth/ba/ddr/experiments/randla/results/debug/queries.pt')

            plotPoints(points.permute(1,0), 'points.ply')
            plotPoints(queries.permute(1,0), 'queries.ply')
            plotPoints(queries.permute(1,0)[:2000],'queries2000.ply')
            plotPoints(points.permute(1,0)[:2000],'points2000.ply')
            plotPoints(inputs['xyz_p'][0][0,:5000], 'pointsTest'+str(0)+'.ply')
            
            for i in range(0,self.num_layers):
                plotPoints(inputs['xyz_p'][i][0], 'points'+str(i)+'.ply')
                plotPoints(self.my_random_sample(inputs['xyz_p'][i][0].permute(1,0).unsqueeze(0).unsqueeze(-1), 
                                                 inputs['sub_idx'][i]).squeeze(0).squeeze(-1).permute(1,0),
                                                 'subpoints'+str(i)+'.ply')
                plotPoints(inputs['xyz_q'][i][0], 'queries'+str(i)+'.ply')
                plotNeighborsPx2(inputs['xyz_p'][i][0],inputs['neigh_idx'][i],
                                 inputs['xyz_p'][i][0],
                                 'neighidx'+str(i)+'.ply')
                plotNeighborsPx2(inputs['xyz_p'][i][0],inputs['interp_skip_idx'][i],
                                 inputs['xyz_q'][i][0],
                                 'interp_skip_idx'+str(i)+'.ply')
                if (i == self.num_layers-1):
                    plotNeighborsPx2(inputs['xyz_p'][-1][0],
                                     inputs['interp_idx'][i],inputs['xyz_q'][i] [0],
                                     'interp_idx'+str(i)+'.ply')
                else:
                    plotNeighborsPx2(inputs['xyz_q'][i+1][0],inputs['interp_idx'][i],
                                     inputs['xyz_q'][i][0],
                                     'interp_idx'+str(i)+'.ply')
            plotPoints(inputs['xyz_p'][i][0][-1], 'points'+str(self.num_layers)+'.ply')
            
            # Printing Shapes
            for i in range(0,self.num_layers):
                print('points: ', inputs['xyz_p'][i].shape)
                print('queries: ', inputs['xyz_q'][i].shape)
                print('neighidx: ', inputs['neigh_idx'][i].shape)
                print('interp_idx: ', inputs['interp_idx'][i].shape)
                print('interp_skip_idx: ', inputs['interp_skip_idx'][i].shape)
                print('---')
            print('=====')
            pdb.set_trace()
            
        ''' PREPARATION '''
        end_points = inputs
        if self.include_normals:
            features = features[:6,:].unsqueeze(0)  # Batch*channel*npoints
        else:
            features = features[:3,:].unsqueeze(0)  # Batch*channel*npoints
        
        if self.debug:
            plotPoints(features[0][:3,:].permute(1,0), 'features.ply')
            
        # Check if too few queries/points (eveyrhing will be zero and we also return zero)
        if (points.shape[1] < 1500 or queries.shape[1] < 1500):
            if self.debug:
                print('less than 1500')
            return torch.zeros(size=(self.latent_size, queries.shape[1]),device='cuda')
        
            
        ''' ReconRLA '''
        features = self.fc0(features)
        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.num_layers):
            if (self.debug):
                self.debugCounter = i
                print('neighidx: ',end_points['neigh_idx'][i].shape)
                print('subidx: ',end_points['sub_idx'][i].shape)
                print('features: ',features.shape)
                print('----')
                
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz_p'][i], end_points['neigh_idx'][i])
            f_sampled_i = self.my_random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################
        
        if self.debug:
            print('=====')
            
        featuers = f_encoder_list[-1]
        if self.enable_middle_mlp:
            features = self.decoder_0(features)
            
        if self.debug:   
            print('features: ',features.shape)
            print('=====')
        
        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.num_layers):
            ### DEBUGGING ###
            if (self.debug):
                print('interp: ',end_points['interp_idx'][-j - 1].shape)
                print('interp_skip: ',end_points['interp_skip_idx'][-j - 1].shape)
                print('features: ',features.shape)
                print('f_enocder_list[',-j - 2,']')
                print('interp_skip_idx[',-j-1,']')
                print('----')
            ### DEBUGGING ###
            
            # Collect features vectors from k-nearest neighbors. (Both from layer below and the skip connection)
            features_layer_below = gather_neighbour(
                features.squeeze(-1).permute(0,2,1), 
                end_points['interp_idx'][-j - 1])
            features_skip_connection = gather_neighbour(
                f_encoder_list[-j - 2].squeeze(-1).permute(0,2,1), 
                end_points['interp_skip_idx'][-j - 1])
            
            # Local Features aggregation
            if j == 0:
                # Case: layer below is from points
                features = self.decoder_blocks2[j](features_layer_below,
                                        features_skip_connection,
                                        end_points['interp_idx'][-j - 1], # neighbour matrix layer below
                                        end_points['interp_skip_idx'][-j - 1], #neighbour matrix skip connection
                                        end_points['xyz_p'][-j - 1], # coordinates points layer below
                                        end_points['xyz_p'][-j - 2], # coordinates points skip connection
                                        end_points['xyz_q'][-j - 1], # coordinates of queries for which we calc. their encoding
                                        end_points['query_neighbors_idx'][-j - 1] # neighbour matrix current layer
                                       )
            else:
                # Case: layer below is from queries
                features = self.decoder_blocks2[j](features_layer_below,
                                        features_skip_connection,
                                        end_points['interp_idx'][-j - 1], # neighbour matrix layer below
                                        end_points['interp_skip_idx'][-j - 1], #neighbour matrix skip connection
                                        end_points['xyz_q'][-j - 1], # coordinates points layer below
                                        end_points['xyz_p'][-j - 2], # coordinates points skip connection
                                        end_points['xyz_q'][-j - 1], # coordinates of queres we calc. their encoding
                                        end_points['query_neighbors_idx'][-j - 1] # neighbour matrix current layer
                                       )
        # ###########################Decoder############################

        if(inputs['permutation'] != []):
            features = features[:,:,torch.argsort(inputs['permutation'])]
        
        features = features.squeeze(0).squeeze(-1) # [latent_size, Q]
        
        if self.debug:
            print('features:', features.shape)
            pdb.set_trace()
            
        return features

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        # Here we subsamble but we don't take the feature encoding of the points we keep,
        # but we keep the encoding that is the maximum of all the enodings of the neighbours of our point
        
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features
    
    @staticmethod
    def my_random_sample(features, pool_idx):
        """
        :param feature: [B, d, N, 1] input features matrix
        :param pool_idx:[B, N', k] Artifact from before
        :return: pool_features = [B, d, N', 1] subsampled feature matrix
        """
        numPoints = pool_idx.shape[1]
        subsampled_features = features.permute(0,2,1,3)[:,:numPoints]
        subsampled_features = subsampled_features.permute(0,2,1,3)
        return subsampled_features

        
class Decoder_lfa(nn.Module):
    def __init__(self, debug, dim_lb, dim_sc, dim_nl, num_agg, agg_k):
        super().__init__()
        self.debug = debug
        self.num_agg = num_agg
        self.agg_k = agg_k

        self.mlpLB = pt_utils.Conv2d(10, 10, kernel_size=(1,1), bn=True)
        self.mlpSC = pt_utils.Conv2d(10, dim_lb+10-dim_sc, kernel_size=(1,1), bn=True)
        self.att_pooling = Att_pooling(dim_lb+10, dim_nl)
        
        if num_agg > 1:
            self.mlpCL = pt_utils.Conv2d(10, 10, kernel_size=(1,1), bn=True)
            self.att_pooling2 = Att_pooling(dim_nl+10,dim_nl)
        
        if self.debug:
            print('MLPSC: ',10,'-',dim_lb+10-dim_sc)
            print('Dec_Att: ',dim_lb+10,'-',dim_nl)
        
    def forward(self, features_lb, features_sc, NM_lb, NM_sc, coords_lb, coords_sc, coords_queries, NM_cl):
        '''
        :param features_lb (features_layerBelow) [B, Q_up, k, d1]
        :param features_sc (featues_skipConnection) [B, Q_up, k, d0]
        :param NM_lb (NeighbourMatrix_layerBelow) [B, Q_up, k]
        :param NM_sc (NeighbourMatrix_skipConnection) [B, Q_up, k]
        :param coords_lb (coordinates_layerBelow) [B, N_down or Q_down, k]
        :param coords_sc (coordinates_skipConnection) [B, N_skip, k]
        :param coords_queries (coordinates_queries) [B, Q_up, k]
        :param NM_cl (NeighbourMatrix_currentLayer) [B, Q_down, k2]
        return
        '''
        if self.debug:
            print('features_lb: ', features_lb.shape)
            print('features_sc: ', features_sc.shape)
            print('NM_lb: ', NM_lb.shape)
            print('NM_sc: ', NM_sc.shape)
            print('coords_lb: ', coords_lb.shape)
            print('coords_sc: ', coords_sc.shape)
            print('coords_queries: ', coords_queries.shape)

        # Relative Point Position Encoding for layer below 
        rppe_lb = dec_relative_pos_encoding(coords_queries,coords_lb,NM_lb)
        rppe_lb = rppe_lb.permute(0,3,1,2) # [B,10,Q_up,k]
        rppe_lb = self.mlpLB(rppe_lb) # [B,d2,Q_up,k]
        
        # Relative Point Position Encoding for skip connection
        rppe_sc = dec_relative_pos_encoding(coords_queries,coords_sc,NM_sc)
        rppe_sc = rppe_sc.permute(0,3,1,2) # [B,10,Q_up,k]
        rppe_sc = self.mlpSC(rppe_sc) # [B,d3,Q_up,k]
        
        # Concatenate RPPE with featuers vector for the layer below
        rppe_lb = rppe_lb.permute(0,2,3,1) # [B,Q_up,k,d2]
        locSE_lb = torch.cat([rppe_lb, features_lb], dim=3) # [B,Q_up,k,d2+d1]
        
        # Concatenate RPPE with featuers vector for the skip connection
        rppe_sc = rppe_sc.permute(0,2,3,1) # [B,Q_up,k,d3]
        locSE_sc = torch.cat([rppe_sc, features_sc], dim=3) # [B,Q_up,k,d3+d0]
        
        # Stack the LocSE of layer below and skip connection to have (2*k)
        f_hat_lb_sc = torch.cat([locSE_lb,locSE_sc], dim=2) # [B,Q_up,2k,d3+d0]
        
        # Attentive Pooling
        f_hat_lb_sc = f_hat_lb_sc.permute(0,3,1,2) # [B,d3+d0,Q_up,2k]
        f_tilde = self.att_pooling(f_hat_lb_sc) # [B,d_out,Q_up,1]
        
        # Second feature aggregation
        if self.num_agg > 1:
            # Relative Point Position Encoding for current layer 
            rppe_cl = dec_relative_pos_encoding(coords_queries,coords_queries,NM_cl)
            rppe_cl = rppe_cl.permute(0,3,1,2) # [B,10,Q_up,k2]
            rppe_cl = self.mlpCL(rppe_cl) # [B,d2,Q_up,k]
            
            # Gather features
            features_current_layer = gather_neighbour(
                f_tilde.permute(0,2,1,3).squeeze(3),
                NM_cl)
            
            # Concatenate RPPE with features
            rppe_cl = rppe_cl.permute(0,2,3,1) # [B,Q_up,k2,10]
            locSE_cl = torch.cat([rppe_cl, features_current_layer], dim=3)
            
            # Attentive Pooling
            locSE_cl = locSE_cl.permute(0,3,1,2)
            f_tilde = self.att_pooling2(locSE_cl)
            
        return f_tilde
        

class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):
        '''
        :param feature_set [B, d_in, N, k]
        :return [B, d_out, N, 1]
        '''
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


def dec_relative_pos_encoding(xyz_newPoints, xyz_sourcePoints, neigh_idx):
    '''
    :param xyz_newPoints (coordinates of the points we want to give a feature encoding)
    :param xyz_sourcePoints (coordinates of the points we already have a featuer encoding)
    :param neigh_idx (neighborMatrix from the newPoints to the soucePoints)
    '''
    
    # Gather xyz for the xyz of the sourcePoint neighbours of the newPoints
    neighbor_xyz = gather_neighbour(xyz_sourcePoints, neigh_idx)
    
    # Gather xyz for the newPoints and reshape 
    xyz_tile = xyz_newPoints.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
    
    # Do calculations (do relative point position encoding)
    relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
    relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
    relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
    
    # Retun 
    return relative_feature


def relative_pos_encoding(xyz, neigh_idx):     
    neighbor_xyz = gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3
    xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
    relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
    relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
    relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
    return relative_feature
    
    
def gather_neighbour(pc, neighbor_idx):
    '''
    What: gather the coordinates or features of neighboring points
    
    :param pc: [B, N, d]
    :param neighbor_idx: [B, N', k]
    :return [B, N', k, d]
    '''
    batch_size = pc.shape[0]
    num_points = neighbor_idx.shape[1]
    d = pc.shape[2]
    index_input = neighbor_idx.reshape(batch_size, -1)
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
    features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
    return features



######### PLOTTING METHODS ##########
import trimesh
import os
def plotPoints(points,filename):
    pc = trimesh.Trimesh(vertices=points.cpu(),
                             faces=None,
                             process=False)
    root_dir = '/home/nicolas/eth/ba/ddr/experiments/randla/results/debug/'
    path = os.path.join(root_dir, filename)
    pc.export(path, vertex_normal=False)

    
def plotNeighborsPx2(points,neighborMatrix,queries,filename):
    '''
    points: [(0.23, -0.33, 0.67)] (N,3)
    NeighborMatrix (B,Q,k) Werte zwischen 0,N-1
    queries: [(0.23, -0.33, 0.67)] (Q,3)
    '''
    # Vertices
    neighbors = points[neighborMatrix[0,:]]
    neighbors = neighbors.reshape(-1,3)
    points2 = torch.repeat_interleave(queries, repeats=neighborMatrix.shape[-1], dim=0)
    vertices = torch.cat([neighbors, points2, points2])
    
    # Faces
    faces = torch.arange(3 * neighbors.shape[0]).reshape(3, -1).t()
    
    pc = trimesh.Trimesh(vertices=vertices.cpu(),
                             faces=faces.cpu(),
                             process=False)
    root_dir = '/home/nicolas/eth/ba/ddr/experiments/randla/results/debug/'
    path = os.path.join(root_dir, filename)
    pc.export(path, vertex_normal=False)
