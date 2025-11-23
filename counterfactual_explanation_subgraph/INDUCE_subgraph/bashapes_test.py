import argparse

from tqdm import tqdm

from cmd_args import get_options
from source.src.chameleon_train import load_GCN_model
from utils.dataloader import DataLoader
from utils.classificationnet import GCNSynthetic
from utils.player import Player
from utils.env import Env
from utils.utils import *
from utils.policynet import *
import numpy as np
import time
import torch
import pickle
from cmd_args import get_options

args = get_options()
args.device = 'cpu'


if(args.device == 'cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1" #set according to gpu avalaibility

switcher = {'gcn':PolicyNetwork, 'gat':PolicyNetwork2, 'sage':PolicyNetwork3}
res_file = open('../logs/{}/log_test_{}_{}.txt'.format(args.dataset, args.dataset, args.save_prefix), 'w')
# PATH = "./saved_models/model_{}.pt".format(args.dataset)

#load model to be explained
def loadModel(args, g):
    if(args.saved):
        model = GCNSynthetic(nfeat=g.feats.shape[1], nhid=args.hidden, nout=args.hidden,
                        nclass=len(g.labels.unique()), dropout=args.dropout)
        model.load_state_dict(torch.load(
            "./models/gcn_3layer_{}.pt".format(args.dataset)))
    else: #train model, save it and return
        data_obj = DataLoader(args.dataset)
        g = data_obj.preprocessData()
        train_model(args, g)

    model.eval()
    output = model(g.feats, g.norm_adj)
    y_pred_orig = torch.argmax(output, dim=1)
    print("y_true counts: {}".format(np.unique(g.labels.numpy(), return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(),
                                                    return_counts=True)))      # Confirm model is actually doing something
    print("Accuracy: ", accuracy(output, g.labels))
    return model

def log_results(cf_dict):
    final_res = open('../results/{}/{}_test_{}.pkl'.format(args.dataset, args.dataset, args.save_prefix),'wb') #no_kl_del_only_inductive overwritten by adaptive beta
    pickle.dump(cf_dict, final_res)
    final_res.close()


def compute_deg_diff(orig_sub_adj, edited_sub_adj):
    orig_degrees = torch.sum(orig_sub_adj)
    new_degrees = torch.sum(edited_sub_adj)
    deg_diff = torch.sum(
        torch.abs(new_degrees - orig_degrees) / (1 + orig_degrees)
    )
    return deg_diff


def compute_motif_viol(orig_sub_adj, edited_sub_adj, tau_c):
    orig_cluster_coef = clustering_coefficient(orig_sub_adj)
    new_cluster_coef = clustering_coefficient(edited_sub_adj)
    motif_violation = torch.sum(
        torch.clamp(torch.abs(new_cluster_coef - orig_cluster_coef) - tau_c, min=0.0)
    )
    return motif_violation


def clustering_coefficient(adj_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Approximate calculation of the local clustering coefficient of an undirected graph using PyTorch (vectorized implementation).
    Note: This is an approximation of the traditional clustering coefficient and is mainly used for training and loss calculation.
    """
    # Calculate the degree of each node
    degrees = torch.sum(adj_tensor, dim=1)

    # Computes A², whose diagonal elements are the number of paths that exist between a node's neighbors (each edge is calculated twice)
    A_squared = torch.mm(adj_tensor, adj_tensor)
    # The actual number of edges between the neighbors of node i is approximately (A_squared[i, i] - degrees[i]) / 2.0
    # Subtract degrees[i] because the adjacency matrix diagonal (self-loop) is also counted and usually needs to be subtracted
    # Simplify the process here and use the diagonal of A_squared directly.
    triangles = torch.diag(A_squared) / 2.0  # More precise calculations may require adjustments

    # Calculate the maximum possible number of edges k*(k-1)/2
    max_possible_edges = degrees * (degrees - 1) / 2.0

    # Avoid dividing by zero: for nodes with degree less than 2, the clustering coefficient is set to 0
    # clustering_coeffs = torch.zeros_like(degrees, dtype=torch.float32)
    # valid_mask = (degrees > 1)
    # clustering_coeffs[valid_mask] = triangles[valid_mask] / max_possible_edges[valid_mask]

    clustering_coeffs = triangles / (max_possible_edges + eps)
    clustering_coeffs = torch.where(degrees > 1, clustering_coeffs, torch.zeros_like(clustering_coeffs))

    return clustering_coeffs



class Eval(object):
    def __init__(self, args):
        self.args = args
        self.dataset = self.args.dataset
        # loading data
        data_obj = DataLoader(args.dataset, args.add_self_loops)  # , args.add_self_loops
        self.graph = data_obj.preprocessData()
        adj, features, labels, pyg_data = None, None, None, None
        if self.dataset == "cora":
            _, adj, features, labels, pyg_data = data_obj.loadCora()
        elif self.dataset == "chameleon":
            _, adj, features, labels, pyg_data = data_obj.loadChameleon()
        elif self.dataset == "BA-SHAPES":
            _, adj, features, labels, pyg_data = data_obj.loadBAShapes()
        else:
            pass
       
        # model whose explanation we will generate
        # self.model = loadModel(self.args, self.graph)
        file_path = f'../models/{test_model}/{dataset_name}/{gcn_layer}-layer/gcn_model.pth'
        self.model = load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay,
                                    with_bias, gcn_layer)
        # self.model = GCNtoPYG(gnn_model, device, features, labels, gcn_layer)

        #explainaing only non-zero and correctly-classified instances
        idx = (self.graph.labels[self.graph.idx_test] > 0).nonzero(as_tuple=False)
        self.targets = self.graph.idx_test[idx] 
        if(self.args.verbose):
                print("Number of non zero instances: ", idx.shape[0])
        
        self.players, self.rshapers = [], []
        self.cf_dict = {}
        self.chosen_targets = []
        self.time_cost_list = []
        i=0
        for t in tqdm(test_nodes, desc="Player Generate for target node"):
            start_time = time.time()
            # p = Player(self.graph, t, self.model, args).cuda()
            p = Player(self.graph, t, self.model, args)
            # if(self.graph.labels[t].to(args.device) == p.orig_out):
            #     self.players.append(p)
            #     # self.chosen_targets.append(t.item())
            #     self.chosen_targets.append(t)
            #     i+=1
            self.players.append(p)
            # self.chosen_targets.append(t.item())
            self.chosen_targets.append(t)
            i += 1
            time_cost = time.time() - start_time
            self.time_cost_list.append(time_cost)

        if(self.args.verbose):
            print("Number of instances: ", len(self.players))
            print('Chosen targets (idx): ', self.chosen_targets)  
        #save eval set indices in pkl file
        eval_set = open('../eval_set/{}.pkl'.format(args.dataset),'wb')
        pickle.dump(self.chosen_targets, eval_set)
        eval_set.close() 

        self.env = Env(self.players, self.args, torch.max(self.graph.labels)+1)
        # self.policy = switcher[args.policynet](args,self.env.statedim).cuda()
        self.policy = switcher[args.policynet](args, self.env.statedim)
        self.policy.load_state_dict(torch.load("../saved_models/{}/model_{}_{}.pt".format(args.dataset, args.dataset, args.save_prefix))['model_state_dict'])

    def policyQuery(self, playerid=0):
        self.playerid = playerid
        self.env.reset(playerid)
        rewards, logp_actions, p_actions = [], [], []
        self.states, self.actions = [], []

        initialrewards=0 #reward at timestep 0
        rewards.append(initialrewards)

        b = self.args.maxbudget 
        orig_out = torch.argmax(self.model(self.players[playerid].G_orig.feats.to(self.args.device), self.players[playerid].G_orig.norm_adj.to(self.args.device))[self.players[playerid].G_orig.target_idx]).item()
        curr_out = torch.argmax(self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]).item()
        while (orig_out == curr_out and b>0 and len(self.players[playerid].cand_dict)>1): 
            state = self.env.getState(playerid) #state = (feats, norm_adj)
            self.states.append(state)
            _, logits = self.policy(state[0].to(self.args.device), state[1].to(self.args.device), self.players[playerid].cand_dict)
            action,logp_action, p_action = self.policy.get_action(logits, self.players[playerid].cand_dict, False)
            
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            reward, loss_pred, loss_graph_dist = self.env.step(action,playerid)
            rewards.append(reward) #send action idx
            curr_out = torch.argmax(self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]).item()
            b-=1

        eval_data = {}
        num_E_plaus = 0
        num_E_minus = 0

        counterfactual = []
        flag = 'not found'
        if(orig_out != curr_out):
            for i in range(len(self.players[playerid].cf_cand)):
                val = [self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][0]], self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][1]], self.players[playerid].cf_cand[i][1]]
                counterfactual.append(val)
                if self.players[playerid].cf_cand[i][1] == "add":
                    num_E_plaus += 1
                else:
                    num_E_minus += 1
            # res_file.write('target: {}, cf: {}\n'.format(self.players[playerid].target.item(), counterfactual)) #self.players[playerid].cf
            res_file.write('target: {}, cf: {}\n'.format(self.players[playerid].target,
                                                         counterfactual))  # self.players[playerid].cf
            # print(self.players[playerid].cf_cand)
            flag = 'found'
            eval_data["success"] = True
            eval_data["target_node"] = self.players[playerid].target
            eval_data["explanation_size"] = len(counterfactual)
            eval_data["added_edges"] = num_E_plaus
            eval_data["removed_edges"] = num_E_minus
            cf_adj = self.players[playerid].G_curr.adj
            sub_adj = self.players[playerid].G_orig.adj
            L_plau = α2 * compute_deg_diff(sub_adj, cf_adj) + α3 * compute_motif_viol(sub_adj, cf_adj, tau_c)
            eval_data["S_plau"] = 2 * (1 - 1 / (1 + torch.exp(-1 * 1.0 * L_plau)))
        else:
            for i in range(len(self.players[playerid].cf_cand)):
                val = [self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][0]], self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][1]], self.players[playerid].cf_cand[i][1]]
                counterfactual.append(val)
            eval_data["success"] = False
            eval_data["target_node"] = self.players[playerid].target
            eval_data["explanation_size"] = 0
            eval_data["added_edges"] = 0
            eval_data["removed_edges"] = 0
            eval_data["S_plau"] = 0

        eval_data["prob_pred_orig"] = self.model(self.players[playerid].G_orig.feats.to(self.args.device),
                                                 self.players[playerid].G_orig.norm_adj.to(self.args.device))[
            self.players[playerid].G_orig.target_idx]
        eval_data["prob_new_actual"] = self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]
        eval_data["time_cost"] = self.time_cost_list[playerid]

        evaluate_data.append(eval_data)

        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        
        return counterfactual, self.players[playerid].G_curr.target_idx.item(),  self.players[playerid].G_curr.adj, self.players[playerid].G_orig.adj, self.players[playerid].G_curr.adj.shape[0], self.players[playerid].G_curr.node_map, orig_out, curr_out, self.players[playerid].G.labels[self.players[playerid].target], reward, loss_pred, loss_graph_dist, flag
    
    def policyQueryRun(self):
        for i, p in tqdm(enumerate(self.players), desc="Player process"):
            cf, new_idx, cf_adj, sub_adj, num_nodes, node_dict ,orig_label, cf_label, label, total_loss, loss_pred, loss_graph_dist, found = self.policyQuery(i)
            # self.cf_dict[p.target.item()] = [new_idx, cf_adj, sub_adj, cf, num_nodes, node_dict ,orig_label, cf_label, label, total_loss, loss_pred, loss_graph_dist, found]
            self.cf_dict[p.target] = [new_idx, cf_adj, sub_adj, cf, num_nodes, node_dict, orig_label, cf_label,
                                             label, total_loss, loss_pred, loss_graph_dist, found]
            
        return self.cf_dict
 

if __name__ == "__main__":
    # --dataset BA-SHAPES --use_onehot --use_degree --use_entropy --ent 0.1 --policynet gat --maxbudget 5 --seed 102 --k 4  --verbose --save_prefix inductive_non0_correct_only
    test_model = "GCN"
    dataset_name = "BA-SHAPES"
    gcn_layer = 2
    nhid = 100
    dropout = 0
    device = "cpu"
    lr = 0.001
    weight_decay = 0.001
    with_bias = True
    test_nodes = [3, 6, 8, 16, 17, 18, 19, 36, 42, 49, 57, 59, 61, 71, 82, 102, 111, 120, 138, 140, 162, 175, 178, 185, 206, 208, 214, 224, 230, 235, 236, 238, 240, 246, 253, 258, 260, 268, 272, 274, 277, 281, 284, 288, 293, 294, 300, 305, 306, 307, 308, 312, 315, 319, 321, 324, 327, 333, 337, 338, 339, 342, 344, 351, 359, 361, 363, 365, 372, 376, 378, 379, 384, 387, 388, 394, 395, 396, 409, 414, 415, 417, 424, 425, 429, 435, 438, 439, 441, 445, 446, 448, 449, 455, 461, 462, 465, 466, 469, 474, 478, 480, 481, 485, 486, 488, 493, 494, 495, 504, 505, 506, 508, 509, 513, 517, 519, 523, 524, 526, 528, 529, 534, 540, 547, 549, 552, 559, 562, 570, 573, 576, 583, 584, 586, 589, 594, 595, 599, 602, 604, 609, 614, 616, 618, 619, 623, 624, 629, 634, 650, 654, 663, 668, 669, 670, 671, 673, 675, 676, 678, 679, 684, 685, 688, 689, 691, 692, 693, 694]
    α2 = 1.5
    α3 = 1.0
    tau_c = 0

    evaluate_data = []
    eval = Eval(args)
    start = time.time()
    cf_dict = eval.policyQueryRun()
    print('Time taken (in sec): ', time.time() - start) 
    log_results(cf_dict)
    res_file.close()

    with open('../results/{}/{}_test_eval.pickle'.format(args.dataset, args.dataset), "wb") as fw:
        pickle.dump(evaluate_data, fw)

    misclas_num = 0
    fidelity = 0.0
    added_edges_num = 0.0
    deleted_edges_num = 0.0
    edited_num = 0.0
    S_plau = 0.0
    time_list = []
    for data in evaluate_data:
        if data["success"]:
            misclas_num += 1
            S_plau += data["S_plau"]

        prob_pred_orig = torch.exp(data["prob_pred_orig"])
        label_pred_orig = data["prob_pred_orig"].argmax().item()
        prob_new_actual = torch.exp(data["prob_new_actual"])
        fidelity += prob_pred_orig[label_pred_orig] - prob_new_actual[label_pred_orig]

        added_edges_num += data["added_edges"]
        deleted_edges_num += data["removed_edges"]
        edited_num += data["added_edges"] + data["removed_edges"]
        time_list.append(data["time_cost"])

    print("Num of target nodes: ", len(test_nodes))
    print("Num of misclassification: ", misclas_num)
    print("Num of cf examples found: {}/{}".format(misclas_num, len(evaluate_data)))
    print("Metric 1 - Misclassification Rate: {:.2f}".format(misclas_num / len(test_nodes)))
    print("Metric 2 - Fidelity: {:.4f}".format(fidelity / len(test_nodes)))
    print("Metric 3 - Average Explanation Size: {:.2f}, E+: {:.2f}, E-: {:.2f}".format(edited_num / misclas_num,
                                                                                       added_edges_num / misclas_num,
                                                                                       deleted_edges_num / misclas_num))
    print("Metric 4 - Average Plausibility: {:.2f}".format(S_plau / misclas_num))
    print("Metric 5 - Average Time Cost: {:.2f}s/per".format(np.mean(np.array(time_list))))
