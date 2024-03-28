import time

import numpy as np
import pandas as pd
import torch
from gymnasium.spaces.graph import GraphInstance
from torch.nn import LeakyReLU, Linear, Parameter, ReLU, Sequential
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
    
class InvariantMLPTianshouPPO_ACTOR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(InvariantMLPTianshouPPO_ACTOR, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        self.lin4 = Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax()
        self.leaky_relu = LeakyReLU()
        
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        picks_left = x["picks_left"].to(device)
        mask = x["mask"].to(device)
        # print(mask)
        data_list = [Data(x=x["graph_nodes"][index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        x = batch.x.float()
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin4(x)
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        return x, state
    
class InvariantMLPTianshouPPO_CRITIC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(InvariantMLPTianshouPPO_CRITIC, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        self.lin4 = Linear(out_channels, 1)
        self.leaky_relu = LeakyReLU()
        
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        picks_left = x["picks_left"].to(device)
        data_list = [Data(x=x["graph_nodes"][index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        x = batch.x.float()
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        x = scatter(src=x, index=batch.batch, dim=0, reduce="sum")
        picks_left = picks_left.unsqueeze(dim=1).float()
        x = self.lin4(x)
        return x
    
class GCNNetTianshouPPO_CRITIC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channnels):
        super(GCNNetTianshouPPO_CRITIC, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(3):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channnels)
        self.lin1_after_scatter = Linear(out_channnels, 1)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        data_list = [Data(x=x["graph_nodes"][index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        x = batch.x.float()
        edge_index = batch.edge_index.long()
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)#x = x.relu()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.leaky_relu(x)#x = x.relu()
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = scatter(src=x, index=batch.batch, dim=0, reduce="sum")
        x = self.lin1_after_scatter(x)
        return x


class GCNNetTianshouPPO_ACTOR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNNetTianshouPPO_ACTOR, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(3):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.lin3 = Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        data_list = [Data(x=x["graph_nodes"][index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        
        mask = x["mask"].to(device)
        edge_index = batch.edge_index.long()
        x = batch.x.float()
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)#x = x.relu()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.leaky_relu(x)#x = x.relu()
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        return x, state
    
    
class GINNetTianshouPPO_CRITIC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINNetTianshouPPO_CRITIC, self).__init__()
        mlp_1 = Sequential(Linear(in_channels, hidden_channels),
                         LeakyReLU(),
                         Linear(hidden_channels, hidden_channels),
                         LeakyReLU())
        self.conv1 = GINConv(mlp_1, train_eps=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(3):
            mlp_inter = Sequential(Linear(hidden_channels, hidden_channels),
                                LeakyReLU(),
                                Linear(hidden_channels, hidden_channels),
                                LeakyReLU())
            self.convs.append(GINConv(mlp_inter, train_eps=True))
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.lin1_after_scatter = Linear(out_channels, 1)
        self.leaky_relu = LeakyReLU()
        
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        data_list = [Data(x=x["graph_nodes"][index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        edge_index = batch.edge_index.long()
        x = batch.x.float()
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = scatter(src=x, index=batch.batch, dim=0, reduce="sum")
        x = self.lin1_after_scatter(x)
        return x

        
class GINNetTianshouPPO_ACTOR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINNetTianshouPPO_ACTOR, self).__init__()
        mlp_1 = Sequential(Linear(in_channels, hidden_channels),
                         LeakyReLU(),
                         Linear(hidden_channels, hidden_channels),
                         LeakyReLU())
        self.conv1 = GINConv(mlp_1, train_eps=True)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(3):
            mlp_inter = Sequential(Linear(hidden_channels, hidden_channels),
                                LeakyReLU(),
                                Linear(hidden_channels, hidden_channels),
                                LeakyReLU())
            self.convs.append(GINConv(mlp_inter, train_eps=True))
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.lin3 = Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.leaky_relu = LeakyReLU()
        
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        data_list = [Data(x=x["graph_nodes"][index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        
        mask = x["mask"].to(device)
        edge_index = batch.edge_index.long()
        x = batch.x.float()
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)#x = x.relu()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.leaky_relu(x)#x = x.relu()
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        return x, state
    

class Inv_MLP_EMBEDDING_Preprocessor(torch.nn.Module):
    def __init__(self):
        super(Inv_MLP_EMBEDDING_Preprocessor, self).__init__()
    
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        picks_left = x["picks_left"].to(device)
        mask = x["mask"].to(device)
        aisle_nrs = x["aisle_nrs"]#.to(device)
        # Add the aisle nrs tot the nodes for batch creation,
        # These are extracted after the batch creation
        x_for_batch = torch.cat((
             x["graph_nodes"],
             aisle_nrs.view(aisle_nrs.shape[0], -1, 1)), dim=-1)
        data_list = [Data(x=x_for_batch[index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device) 
        # Separate aisle nrs from the node features again
        aisle_nrs = batch.x[:, -1].to(torch.int64)
        x = batch.x[:, :-1].float()
        return x, batch, aisle_nrs, picks_left, mask, batch_size
        

class INV_MLP_EMBEDDING_FORWARD(torch.nn.Module):
    def __init__(self, lin1, lin2, lin3, after_emb_lin1, after_emb_lin2, after_emb_lin3,
                 softmax):
        super(INV_MLP_EMBEDDING_FORWARD, self).__init__()
        self.lin1 = lin1
        self.lin2 = lin2
        self.lin3 = lin3
        self.after_emb_lin1 = after_emb_lin1
        self.after_emb_lin2 = after_emb_lin2
        self.after_emb_lin3 = after_emb_lin3
        self.softmax = softmax
        self.leaky_relu = LeakyReLU()
        
    def forward(self, x, aisle_nrs=None, batch=None, picks_left=None):
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        aisle_ids = aisle_nrs + batch * (aisle_nrs.max() + 1)
        aisle_embeddings = scatter(src=x, index=aisle_ids, dim=0, reduce="mean")
        x = torch.cat((x, aisle_embeddings[aisle_ids.long()]), dim=1)
        x = self.after_emb_lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.after_emb_lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.after_emb_lin3(x)
        return x


class InvariantMLP_WITH_EMBEDDINNGTianshouPPO_ACTOR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_channels,
                 hidden_after_emb, out_channels):
        super(InvariantMLP_WITH_EMBEDDINNGTianshouPPO_ACTOR, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, emb_channels)
        self.after_emb_lin1 = Linear(emb_channels * 2, hidden_after_emb)
        self.after_emb_lin2 = Linear(hidden_after_emb, out_channels)
        self.after_emb_lin3 = Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax()
        self.preprocessor = Inv_MLP_EMBEDDING_Preprocessor()
        self.forward_model = INV_MLP_EMBEDDING_FORWARD(self.lin1, self.lin2, self.lin3,
                                                        self.after_emb_lin1,
                                                        self.after_emb_lin2,
                                                        self.after_emb_lin3,
                                                        self.softmax)
        
        
    def forward(self, x, state=None, info={}):
        t = time.time()
        x, batch, aisle_nrs, picks_left, mask, batch_size = self.preprocessor(x)
        x = self.forward_model(x, aisle_nrs, batch.batch, picks_left)
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        
        t_2 = time.time()
        with open("time.txt", "a") as f:
            f.write(f"{t_2 - t}\n")
        return x, state
    
class InvariantMLP_WITH_EMBEDDINGTianshouPPO_CRITIC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_channels, 
                 hidden_after_emb, out_channels):
        super(InvariantMLP_WITH_EMBEDDINGTianshouPPO_CRITIC, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, emb_channels)
        self.after_emb_lin1 = Linear(emb_channels * 2, hidden_channels)
        self.after_emb_lin2 = Linear(hidden_channels, out_channels)
        self.lin4 = Linear(out_channels, 1)
        self.leaky_relu = LeakyReLU()
        
    def forward(self, x, state=None, info={}):
        # print(x)
        batch_size = x["graph_edges"].shape[0]
        picks_left = x["picks_left"].to(device)
        aisle_nrs = x["aisle_nrs"]
        x_for_batch = torch.cat((
            x["graph_nodes"],
            aisle_nrs.view(aisle_nrs.shape[0], -1, 1)), dim=-1)
        data_list = [Data(x=x_for_batch[index],
                        edge_index=x["graph_edge_links"][index])
                    for index in range(batch_size)]
        
        batch = Batch.from_data_list(data_list).to(device)
        aisle_nrs = batch.x[:, -1].to(torch.int64)
        x = batch.x[:, :-1].float()
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        aisle_ids = aisle_nrs + batch.batch * (aisle_nrs.max() + 1)
        aisle_embeddings = scatter(src=x, index=aisle_ids, dim=0, reduce="mean")
        x = torch.cat((x, aisle_embeddings[aisle_ids]), dim=1)
        x = self.after_emb_lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.after_emb_lin2(x)
        x = scatter(src=x, index=batch.batch, dim=0, reduce="sum")
        x = self.lin4(x)
        return x
    
    
class GINNet_WITH_EMBEDDINGTianshouPPO_ACTOR(torch.nn.Module):
    ### NOTE: Not used
    def __init__(self, in_channels, hidden_channels, emb_channels,
                 hidden_after_emb, out_channels):
        super(GINNet_WITH_EMBEDDINGTianshouPPO_ACTOR, self).__init__()
        mlp_1 = Sequential(Linear(in_channels, hidden_channels),
                         LeakyReLU(),
                         Linear(hidden_channels, hidden_channels),
                         LeakyReLU())
        mlp_inter = Sequential(Linear(hidden_channels, hidden_channels),
                               LeakyReLU(),
                               Linear(hidden_channels, hidden_channels),
                               LeakyReLU())
        mlp_final = Sequential(Linear(hidden_channels, hidden_channels),
                               LeakyReLU(),
                               Linear(hidden_channels, out_channels),
                               LeakyReLU())
        self.conv1 = GINConv(mlp_1, train_eps=True)
        self.conv2 = GINConv(mlp_inter, train_eps=True)
        # self.conv3 = GINConv(mlp_inter, train_eps=True)
        # self.conv4 = GINConv(mlp_inter, train_eps=True)
        self.conv5 = GINConv(mlp_final, train_eps=True)
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv5 = GATConv(hidden_channels, out_channels)
        
        # self.lin1 = Linear(out_channels + 1, 1)
        self.after_emb_lin1 = Linear(emb_channels * 2, hidden_after_emb)
        self.after_emb_lin2 = Linear(hidden_after_emb, out_channels)
        self.after_emb_lin3 = Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.leaky_relu = LeakyReLU()
        # self.conv6 = GINConv(mlp_inter, train_eps=True)
        # self.conv7 = GINConv(mlp_inter, train_eps=True)
        # self.conv8 = GINConv(mlp_inter, train_eps=True)
        # self.conv9 = GINConv(mlp_inter, train_eps=True)
        # self.conv10 = GINConv(mlp_final, train_eps=True)
        
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        picks_left = x["picks_left"].to(device)
        aisle_nrs = x["aisle_nrs"]
        x_for_batch = torch.cat((
            x["graph_nodes"],
            aisle_nrs.view(aisle_nrs.shape[0], -1, 1)), dim=-1)
        data_list = [Data(x=x_for_batch[index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        
        mask = x["mask"].to(device)
        edge_index = batch.edge_index.long()
        aisle_nrs = batch.x[:, -1].to(torch.int64)
        x = batch.x[:, :-1].float()
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv5(x, edge_index)
        aisle_ids = aisle_nrs + batch.batch * (aisle_nrs.max() + 1)
        aisle_embeddings = scatter(src=x, index=aisle_ids, dim=0, reduce="mean")
        x = torch.cat((x, aisle_embeddings[aisle_ids]), dim=1)
        
        x = self.after_emb_lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.after_emb_lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.after_emb_lin3(x)
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        return x, state

    
class Multi_objective_Invariant_MLP_CRITIC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Multi_objective_Invariant_MLP_CRITIC, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        self.lin4 = Linear(out_channels, 2)
        self.leaky_relu = LeakyReLU()
        
    def forward(self, x, state=None, info={}):
        # print(x)
        if isinstance(x["graph"], GraphInstance):
            batch_size = 1
        else:
            batch_size = len(x["graph"])
        picks_left = torch.tensor(x["picks_left"]).to(device).view(batch_size)
        if isinstance(x["graph"], GraphInstance):         
            node_info = torch.from_numpy(x["graph"].nodes)
            node_info = node_info.view(batch_size,
                                       x["graph"].nodes.shape[0],
                                       x["graph"].nodes.shape[1])
        else:
            node_info = torch.from_numpy(np.array([graph.nodes for graph in x["graph"]]))
        if isinstance(x["graph"], GraphInstance):
            edge_indices = torch.from_numpy(x["graph"].edge_links.T).view(
                batch_size,
                x["graph"].edge_links.T.shape[0],
                x["graph"].edge_links.T.shape[1])
        else:
            edge_indices = torch.from_numpy(np.array([graph.edge_links.T for graph in x["graph"]]))

        data_list = [Data(x=node_info[index],
                          edge_index=edge_indices[index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        x = batch.x.float()
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        x = scatter(src=x, index=batch.batch, dim=0, reduce="sum")

        x = self.lin4(x)
        return x
    
class Multi_objective_InvariantMLP_WITH_EMBEDDINNGPPO_ACTOR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_channels,
                 hidden_after_emb, out_channels):
        super(Multi_objective_InvariantMLP_WITH_EMBEDDINNGPPO_ACTOR, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, emb_channels)
        self.after_emb_lin1 = Linear(emb_channels * 2, hidden_after_emb)
        self.after_emb_lin2 = Linear(hidden_after_emb, out_channels)
        self.after_emb_lin3 = Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax()
        self.leaky_relu = LeakyReLU()
        
    def forward(self, x, state=None, info={}):
        if isinstance(x["graph"], GraphInstance):
            batch_size = 1
        else:
            batch_size = len(x["graph"])
        picks_left = torch.tensor(x["picks_left"]).to(device)
        mask = torch.from_numpy(np.array(x["mask"])).to(device)
        aisle_nrs = torch.from_numpy(np.array(x["Aisle_nrs"]))#.to(device)
        # Add the aisle nrs tot the nodes for batch creation,
        # These are extracted after the batch creation
        if isinstance(x["graph"], GraphInstance):         
            node_info = torch.from_numpy(x["graph"].nodes)
            node_info = node_info.view(batch_size,
                                       x["graph"].nodes.shape[0],
                                       x["graph"].nodes.shape[1])
        else:
            node_info = torch.from_numpy(np.array([graph.nodes for graph in x["graph"]]))
        x_for_batch = torch.cat((
             node_info,
             aisle_nrs.view(batch_size, -1, 1)), dim=-1)
        if isinstance(x["graph"], GraphInstance):
            edge_indices = torch.from_numpy(x["graph"].edge_links.T).view(
                batch_size,
                x["graph"].edge_links.T.shape[0],
                x["graph"].edge_links.T.shape[1])
        else:
            edge_indices = torch.from_numpy(np.array([graph.edge_links.T for graph in x["graph"]]))
        data_list = [Data(x=x_for_batch[index],
                          edge_index=edge_indices[index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device) 
        # Separate aisle nrs from the node features again
        aisle_nrs = batch.x[:, -1].to(torch.int64)
        x = batch.x[:, :-1].float()
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        aisle_ids = aisle_nrs + batch.batch * (aisle_nrs.max() + 1)
        aisle_embeddings = scatter(src=x, index=aisle_ids, dim=0, reduce="mean")
        x = torch.cat((x, aisle_embeddings[aisle_ids]), dim=1)
        
        x = self.after_emb_lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.after_emb_lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.after_emb_lin3(x)
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        if torch.isnan(x).any():
            print("Here")
            print(f"{mask=}")
        return x, state
    
class invariant_embedding(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_channels,
                 hidden_after_emb, out_channels):
        super(invariant_embedding, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, emb_channels)
        self.after_emb_lin1 = Linear(emb_channels * 2, hidden_after_emb)
        self.after_emb_lin2 = Linear(hidden_after_emb, out_channels)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x, aisle_nrs, batch):
        # Separate aisle nrs from the node features again
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        aisle_ids = aisle_nrs + batch * (aisle_nrs.max() + 1)
        aisle_embeddings = scatter(src=x, index=aisle_ids, dim=0, reduce="mean")
        x = torch.cat((x, aisle_embeddings[aisle_ids]), dim=1)
        
        x = self.after_emb_lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.after_emb_lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        return x
class invariantMLP_WITH_EMBEDDING_PER_CLASS_TIANSHOU_ACTOR(torch.nn.Module):
    def __init__(self, in_channels_fair, in_channels_efficient,
                 hidden_channels, emb_channels, hidden_after_emb, out_channels):
        super(invariantMLP_WITH_EMBEDDING_PER_CLASS_TIANSHOU_ACTOR, self).__init__()
        self.in_channels_fair = in_channels_fair
        self.in_channels_efficient = in_channels_efficient
        self.embedding_fair = invariant_embedding(in_channels_fair, hidden_channels,
                                                  emb_channels, hidden_channels,
                                                  out_channels)
        self.embedding_perf = invariant_embedding(in_channels_efficient, hidden_channels,
                                                  emb_channels, hidden_channels,
                                                  out_channels)
        self.lin1 = Linear(int(out_channels*2), out_channels)
        self.lin2 = Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax()
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x, state=None, info={}):
        t = time.time()
        batch_size = x["graph_edges"].shape[0]
        picks_left = x["picks_left"].to(device)
        mask = x["mask"].to(device)
        aisle_nrs = x["aisle_nrs"]#.to(device)
        # Add the aisle nrs tot the nodes for batch creation,
        # These are extracted after the batch creation
        x_for_batch = torch.cat((
             x["graph_nodes"],
             aisle_nrs.view(aisle_nrs.shape[0], -1, 1)), dim=-1)
        data_list = [Data(x=x_for_batch[index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        
        # Separate aisle nrs from the node features again
        aisle_nrs = batch.x[:, -1].to(torch.int64)
        x_perf = batch.x[:, :self.in_channels_efficient].float()
        x_fair = batch.x[:, self.in_channels_efficient:-1].float()
        x_perf = self.embedding_perf(x_perf, aisle_nrs, batch.batch)
        x_fair = self.embedding_fair(x_fair, aisle_nrs, batch.batch)
        x = torch.cat((x_perf, x_fair), dim=1)
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        t_2 = time.time()
        with open("time.txt", "a") as f:
            f.write(f"{t_2 - t}\n")
        return x, state

class invariant_critic_embedding(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(invariant_critic_embedding, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin3(x)
        return x
class InvariantMLPTianshou_PER_CLASS_PPO_CRITIC(torch.nn.Module):
    def __init__(self, in_channels_fair, in_channels_efficient,
                 hidden_channels, out_channels):
        super(InvariantMLPTianshou_PER_CLASS_PPO_CRITIC, self).__init__()
        self.in_channels_fair = in_channels_fair
        self.in_channels_efficient = in_channels_efficient
        self.embedding_fair = invariant_critic_embedding(in_channels_fair, hidden_channels,
                                                  out_channels,)
        self.embedding_perf = invariant_critic_embedding(in_channels_efficient, hidden_channels,
                                                  out_channels,)
        self.lin1 = Linear(int(out_channels*2), out_channels)
        self.lin2 = Linear(out_channels, 1)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        picks_left = x["picks_left"].to(device)
        data_list = [Data(x=x["graph_nodes"][index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        x_perf = batch.x[:, :self.in_channels_efficient].float()
        x_fair = batch.x[:, self.in_channels_efficient:].float()
        x_perf = self.embedding_perf(x_perf)
        x_fair = self.embedding_fair(x_fair)
        x = torch.cat((x_perf, x_fair), dim=1)
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = scatter(src=x, index=batch.batch, dim=0, reduce="sum")
        x = self.lin2(x)
        return x
    

class InvariantMLPTianshou_PER_CLASS_PPO_ACTOR(torch.nn.Module):
    def __init__(self, in_channels_fair, in_channels_efficient,
                 hidden_channels, out_channels):
        super(InvariantMLPTianshou_PER_CLASS_PPO_ACTOR, self).__init__()
        self.in_channels_fair = in_channels_fair
        self.in_channels_efficient = in_channels_efficient
        self.embedding_fair = invariant_critic_embedding(in_channels_fair, hidden_channels,
                                                  out_channels,)
        self.embedding_perf = invariant_critic_embedding(in_channels_efficient, hidden_channels,
                                                  out_channels,)
        self.lin1 = Linear(int(out_channels*2), out_channels)
        self.lin2 = Linear(out_channels, 1)
        self.leaky_relu = LeakyReLU()
        self.softmax = torch.nn.Softmax()   
        
    def forward(self, x, state=None, info={}):
        batch_size = x["graph_edges"].shape[0]
        picks_left = x["picks_left"].to(device)
        mask = x["mask"].to(device)
        
        data_list = [Data(x=x["graph_nodes"][index],
                          edge_index=x["graph_edge_links"][index])
                     for index in range(batch_size)]
        
        batch = Batch.from_data_list(data_list).to(device)
        x_perf = batch.x[:, :self.in_channels_efficient].float()
        x_fair = batch.x[:, self.in_channels_efficient:].float()
        x_perf = self.embedding_perf(x_perf)
        x_fair = self.embedding_fair(x_fair)   
        x = torch.cat((x_perf, x_fair), dim=1)
        x = self.lin1(x)
        x = self.leaky_relu(x)#x = x.relu()
        x = self.lin2(x)
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        return x, state

