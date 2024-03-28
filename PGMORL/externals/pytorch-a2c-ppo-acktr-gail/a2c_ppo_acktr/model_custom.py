import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, FixedCategorical
from a2c_ppo_acktr.utils import init

from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter
from torch.nn import LeakyReLU
import pandas as pd
# device = "cuda" if torch.cuda.is_available() else "cpu"

class invariant_embedding(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_channels,
                 hidden_after_emb, out_channels):
        super(invariant_embedding, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, emb_channels)
        self.after_emb_lin1 = nn.Linear(emb_channels * 2, hidden_after_emb)
        self.after_emb_lin2 = nn.Linear(hidden_after_emb, out_channels)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x, aisle_nrs, batch):
        # Separate aisle nrs from the node features again
        x = self.lin1(x)
        x = self.leaky_relu(x)
        x = self.lin2(x)
        x = self.leaky_relu(x)
        x = self.lin3(x)
        # print(x.shape, picks_left.shape)
        # print(f"{batch.batch=}")
        # NOTE TO SELF: Nee to match the picks left to the correct graph/batch and
        # match the sizes of the vector
        aisle_ids = aisle_nrs + batch * (aisle_nrs.max() + 1)
        aisle_embeddings = scatter(src=x, index=aisle_ids, dim=0, reduce="mean")
        # scatter(src=x, index=batch, dim=0, reduce="sum")
        x = torch.cat((x, aisle_embeddings[aisle_ids]), dim=1)
        
        # picks_left = torch.gather(picks_left, 0, batch.batch).unsqueeze(dim=1).float()
        # x = torch.cat((x, picks_left), dim=1)
        x = self.after_emb_lin1(x)
        x = self.leaky_relu(x)
        x = self.after_emb_lin2(x)
        x = self.leaky_relu(x)
        return x
class invariantMLP_WITH_EMBEDDING_PER_CLASS_ACTOR(torch.nn.Module):
    def __init__(self, in_channels_fair, in_channels_efficient,
                 hidden_channels, emb_channels, hidden_after_emb, out_channels):
        super(invariantMLP_WITH_EMBEDDING_PER_CLASS_ACTOR, self).__init__()
        self.in_channels_fair = in_channels_fair
        self.in_channels_efficient = in_channels_efficient
        self.embedding_fair = invariant_embedding(in_channels_fair, hidden_channels,
                                                  emb_channels, hidden_channels,
                                                  out_channels)
        self.embedding_perf = invariant_embedding(in_channels_efficient, hidden_channels,
                                                  emb_channels, hidden_channels,
                                                  out_channels)
        self.lin1 = nn.Linear(int(out_channels*2), out_channels)
        self.lin2 = nn.Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x, state=None, info={}):
        device = next(self.parameters()).device
        assert isinstance(x, (tuple, dict, np.ndarray))
        # assert len(x) == 2 if isinstance(x, tuple) else len(x) == 4
        if isinstance(x, (tuple, dict)):
            batch_size = 1
        else:
            batch_size = x.shape[0]
        # batch_size = x["graph_edges"].shape[0]
        # picks_left = x["picks_left"].to(device)
        if isinstance(x, tuple):
            picks_left = torch.tensor(x[0]["picks_left"]).to(device)
            mask = torch.from_numpy(x[0]["mask"]).to(device)
            aisle_nrs = torch.from_numpy(x[0]["Aisle_nrs"])#.to(device)
            node_info = torch.from_numpy(x[0]["graph"].nodes).view(
                batch_size, x[0]["graph"].nodes.shape[0], -1)
            edge_indices = torch.from_numpy(x[0]["graph"].edge_links.T).view(
                batch_size, x[0]["graph"].edge_links.T.shape[0], -1)
        elif isinstance(x, dict):
            picks_left = torch.tensor(x["picks_left"]).to(device)
            mask = torch.from_numpy(x["mask"]).to(device)
            aisle_nrs = torch.from_numpy(x["Aisle_nrs"])
            node_info = torch.from_numpy(x["graph"].nodes).view(
                batch_size, x["graph"].nodes.shape[0], -1)
            edge_indices = torch.from_numpy(x["graph"].edge_links.T).view(
                batch_size, x["graph"].edge_links.T.shape[0], -1)
        else:
            mask = torch.from_numpy(np.array([d["mask"] if isinstance(
                d, dict) else d[0]["mask"] for d in x[:, 0]])).to(device)
            aisle_nrs = torch.from_numpy(np.array([d["Aisle_nrs"] if isinstance(
                d, dict) else d[0]["Aisle_nrs"] for d in x[:, 0]]))
            node_info = torch.from_numpy(np.array([d["graph"].nodes if isinstance(
                d, dict) else d[0]["graph"].nodes for d in x[:, 0]]))
            edge_ind_shape = x[0, 0]["graph"].edge_links.T.shape \
                if isinstance(x[0, 0], dict) \
                else x[0, 0][0]["graph"].edge_links.T.shape
            edge_indices = torch.from_numpy(
                np.array([d["graph"].edge_links.T if isinstance(
                    d, dict) else d[0]["graph"].edge_links.T for d in x[:, 0]])).view(
                        batch_size, edge_ind_shape[0], -1)
        # print(mask)
        # Add the aisle nrs tot the nodes for batch creation,
        # These are extracted after the batch creation
        x_for_batch = torch.cat((
             node_info,
             aisle_nrs.view(batch_size, -1, 1)), dim=-1)
        data_list = [Data(x=x_for_batch[index],
                          edge_index=edge_indices[index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        
        # Separate aisle nrs from the node features again
        aisle_nrs = batch.x[:, -1].to(torch.int64)
        x_perf = batch.x[:, :self.in_channels_efficient]#.float()
        x_fair = batch.x[:, self.in_channels_efficient:-1]#.float()
        x_perf = self.embedding_perf(x_perf, aisle_nrs, batch.batch)
        x_fair = self.embedding_fair(x_fair, aisle_nrs, batch.batch)
        x = torch.cat((x_perf, x_fair), dim=1)
        x = self.lin1(x)
        x = self.leaky_relu(x)
        x = self.lin2(x)
        
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        return x
class invariant_critic_embedding(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(invariant_critic_embedding, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.leaky_relu(x)
        x = self.lin2(x)
        x = self.leaky_relu(x)
        x = self.lin3(x)
        return x

# @np.vectorize
# def func_to_serialize(x):
#     return x.get("graph")
class InvariantMLP_PER_CLASS_PPO_CRITIC(torch.nn.Module):
    def __init__(self, in_channels_fair, in_channels_efficient,
                 hidden_channels, out_channels):
        super(InvariantMLP_PER_CLASS_PPO_CRITIC, self).__init__()
        self.in_channels_fair = in_channels_fair
        self.in_channels_efficient = in_channels_efficient
        self.embedding_fair = invariant_critic_embedding(in_channels_fair, hidden_channels,
                                                  out_channels,)
        self.embedding_perf = invariant_critic_embedding(in_channels_efficient, hidden_channels,
                                                  out_channels,)
        self.lin1 = nn.Linear(int(out_channels*2), out_channels)
        self.lin2 = nn.Linear(out_channels, 2)
        self.leaky_relu = LeakyReLU()
        # self.vectorized_extraction_func = np.vectorize(lambda x: x.get("graph"))
    
    def forward(self, x, state=None, info={}):
        device = next(self.parameters()).device
        assert isinstance(x, (tuple, dict, np.ndarray))
        # assert len(x) == 2 if isinstance(x, tuple) else len(x) == 4
        if isinstance(x, (tuple, dict)):
            batch_size = 1
        else:
            batch_size = x.shape[0]
        # batch_size = x["graph_edges"].shape[0]
        # picks_left = x["picks_left"].to(device)
        if isinstance(x, tuple):
            picks_left = torch.tensor(x[0]["picks_left"]).to(device)
            node_info = torch.from_numpy(x[0]["graph"].nodes).view(
                batch_size, x[0]["graph"].nodes.shape[0], -1)
            edge_indices = torch.from_numpy(x[0]["graph"].edge_links.T).view(
                batch_size, x[0]["graph"].edge_links.T.shape[0], -1)
        elif isinstance(x, dict):
            picks_left = torch.tensor(x["picks_left"]).to(device)
            node_info = torch.from_numpy(x["graph"].nodes).view(
                batch_size, x["graph"].nodes.shape[0], -1)
            edge_indices = torch.from_numpy(x["graph"].edge_links.T).view(
                batch_size, x["graph"].edge_links.T.shape[0], -1)
        else:
            edge_ind_shape = x[0, 0]["graph"].edge_links.T.shape \
                if isinstance(x[0, 0], dict) \
                    else x[0, 0][0]["graph"].edge_links.T.shape
            node_info = torch.from_numpy(np.array(
                [d["graph"].nodes if isinstance(d, dict) else d[0]["graph"].nodes
                 for d in x[:, 0]]))
            edge_indices = torch.from_numpy(np.array(
                [d["graph"].edge_links.T if isinstance(d, dict) else d[0]["graph"].edge_links.T
                 for d in x[:, 0]])).view(
                batch_size, edge_ind_shape[0], -1)
        data_list = [Data(x=node_info[index],
                          edge_index=edge_indices[index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        x_perf = batch.x[:, :self.in_channels_efficient]#.float()
        x_fair = batch.x[:, self.in_channels_efficient:]#.float()
        x_perf = self.embedding_perf(x_perf)
        x_fair = self.embedding_fair(x_fair)
        x = torch.cat((x_perf, x_fair), dim=1)
        x = self.lin1(x)
        x = self.leaky_relu(x)
        x = scatter(src=x, index=batch.batch, dim=0, reduce="sum")
        x = self.lin2(x)
        return x
    
class CREATE_SUPERVISED_OUTPUTinvariantMLP_WITH_EMBEDDING_PER_CLASS_ACTOR(torch.nn.Module):
    def __init__(self, in_channels_fair, in_channels_efficient,
                 hidden_channels, emb_channels, hidden_after_emb, out_channels):
        super(CREATE_SUPERVISED_OUTPUTinvariantMLP_WITH_EMBEDDING_PER_CLASS_ACTOR, self).__init__()
        self.in_channels_fair = in_channels_fair
        self.in_channels_efficient = in_channels_efficient
        self.embedding_fair = invariant_embedding(in_channels_fair, hidden_channels,
                                                  emb_channels, hidden_channels,
                                                  out_channels)
        self.embedding_perf = invariant_embedding(in_channels_efficient, hidden_channels,
                                                  emb_channels, hidden_channels,
                                                  out_channels)
        self.lin1 = nn.Linear(int(out_channels*2), out_channels)
        self.lin2 = nn.Linear(out_channels, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.leaky_relu = LeakyReLU()
    
    def forward(self, x, state=None, info={}):
        device = next(self.parameters()).device
        assert isinstance(x, (tuple, dict, np.ndarray))
        # assert len(x) == 2 if isinstance(x, tuple) else len(x) == 4
        if isinstance(x, (tuple, dict)):
            batch_size = 1
        else:
            batch_size = x.shape[0]
        # batch_size = x["graph_edges"].shape[0]
        # picks_left = x["picks_left"].to(device)
        if isinstance(x, tuple):
            picks_left = torch.tensor(x[0]["picks_left"]).to(device)
            mask = torch.from_numpy(x[0]["mask"]).to(device)
            aisle_nrs = torch.from_numpy(x[0]["Aisle_nrs"])#.to(device)
            node_info = torch.from_numpy(x[0]["graph"].nodes).view(
                batch_size, x[0]["graph"].nodes.shape[0], -1)
            edge_indices = torch.from_numpy(x[0]["graph"].edge_links.T).view(
                batch_size, x[0]["graph"].edge_links.T.shape[0], -1)
        elif isinstance(x, dict):
            picks_left = torch.tensor(x["picks_left"]).to(device)
            mask = torch.from_numpy(x["mask"]).to(device)
            aisle_nrs = torch.from_numpy(x["Aisle_nrs"])
            node_info = torch.from_numpy(x["graph"].nodes).view(
                batch_size, x["graph"].nodes.shape[0], -1)
            edge_indices = torch.from_numpy(x["graph"].edge_links.T).view(
                batch_size, x["graph"].edge_links.T.shape[0], -1)
        else:
            mask = torch.from_numpy(np.array([d["mask"] if isinstance(
                d, dict) else d[0]["mask"] for d in x[:, 0]])).to(device)
            aisle_nrs = torch.from_numpy(np.array([d["Aisle_nrs"] if isinstance(
                d, dict) else d[0]["Aisle_nrs"] for d in x[:, 0]]))
            node_info = torch.from_numpy(np.array([d["graph"].nodes if isinstance(
                d, dict) else d[0]["graph"].nodes for d in x[:, 0]]))
            edge_ind_shape = x[0, 0]["graph"].edge_links.T.shape \
                if isinstance(x[0, 0], dict) \
                else x[0, 0][0]["graph"].edge_links.T.shape
            edge_indices = torch.from_numpy(
                np.array([d["graph"].edge_links.T if isinstance(
                    d, dict) else d[0]["graph"].edge_links.T for d in x[:, 0]])).view(
                        batch_size, edge_ind_shape[0], -1)
        # print(mask)
        # Add the aisle nrs tot the nodes for batch creation,
        # These are extracted after the batch creation
        x_for_batch = torch.cat((
             node_info,
             aisle_nrs.view(batch_size, -1, 1)), dim=-1)
        data_list = [Data(x=x_for_batch[index],
                          edge_index=edge_indices[index])
                     for index in range(batch_size)]
        batch = Batch.from_data_list(data_list).to(device)
        
        # Separate aisle nrs from the node features again
        aisle_nrs = batch.x[:, -1].to(torch.int64)
        x_perf = batch.x[:, :self.in_channels_efficient]#.float()
        x_fair = batch.x[:, self.in_channels_efficient:-1]#.float()
        x_perf = self.embedding_perf(x_perf, aisle_nrs, batch.batch)
        x_fair = self.embedding_fair(x_fair, aisle_nrs, batch.batch)
        x = torch.cat((x_perf, x_fair), dim=1)
        x = self.lin1(x)
        x = self.leaky_relu(x)
        x = self.lin2(x)
        
        non_zero_mask_indices = mask.flatten().nonzero().flatten()
        selected_y_value = x[non_zero_mask_indices]
        selected_x_value = batch.x[:, :-1][non_zero_mask_indices]
        df_X = pd.DataFrame(selected_x_value.cpu().numpy())
        df_y = pd.DataFrame(selected_y_value.cpu().numpy())
        df_X.to_csv("X.csv", index=False, header=False, mode="a")
        df_y.to_csv("y.csv", index=False, header=False, mode="a")
        
        
        x = x.view(batch_size, -1)
        x = torch.where(mask != 0, x, float("-inf"))
        x = self.softmax(x)
        return x