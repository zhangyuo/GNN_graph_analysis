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
    使用 PyTorch 近似计算无向图的局部聚类系数（向量化实现）。
    注意：这是对传统聚类系数的一种近似，主要用于训练和损失计算。
    """
    # 计算每个节点的度
    degrees = torch.sum(adj_tensor, dim=1)

    # 计算 A²，其对角线元素是节点邻居之间存在的路径数（每条边被计算两次）
    A_squared = torch.mm(adj_tensor, adj_tensor)
    # 节点i的邻居之间实际存在的边数近似为 (A_squared[i, i] - degrees[i]) / 2.0
    # 减 degrees[i] 是因为邻接矩阵对角线（自环）也被计算在内，通常需要减去
    # 这里简化处理，直接使用 A_squared 的对角线
    triangles = torch.diag(A_squared) / 2.0  # 更精确的计算可能需要调整

    # 计算可能存在的最大边数 k*(k-1)/2
    max_possible_edges = degrees * (degrees - 1) / 2.0

    # 避免除以零：对于度小于2的节点，聚类系数设为0
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
        elif self.dataset == "TREE-CYCLES":
            _, adj, features, labels, pyg_data = data_obj.loadTreeCycles()
        elif self.dataset == "Loan-Decision":
            _, adj, features, labels, pyg_data = data_obj.loadLoanDecision()
        elif self.dataset == "ogbn-arxiv":
            _, adj, features, labels, pyg_data = data_obj.loadOgbnArxiv()
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
    # --dataset ogbn-arxiv --use_onehot --use_degree --use_entropy --ent 0.1 --policynet gat --maxbudget 5 --seed 102 --k 3  --verbose --save_prefix inductive_non0_correct_only
    test_model = "GCN"
    dataset_name = "ogbn-arxiv"
    gcn_layer = 2
    nhid = 64
    dropout = 0.5
    device = "cpu"
    lr = 0.01
    weight_decay = 5e-4
    with_bias = True
    test_nodes = [696, 732, 733, 785, 789, 877, 1113, 1131, 1148, 1164, 1219, 1261, 1321, 1944, 1991, 2425, 2475, 2503, 2557, 2584, 2597, 2615, 3332, 3374, 3389, 3401, 3579, 3718, 3742, 3782, 3881, 3891, 4657, 4961, 4962, 4965, 4980, 4987, 5086, 5104, 5135, 5148, 5871, 5875, 5883, 5919, 5956, 6258, 6282, 6391, 6399, 6404, 7179, 7340, 7510, 7630, 7643, 7644, 8435, 8800, 8822, 8830, 8986, 9758, 9768, 9782, 9783, 10179, 10199, 10207, 10245, 11031, 11050, 11089, 11433, 11445, 11475, 11491, 11511, 11581, 12427, 12443, 12468, 12954, 12972, 13014, 13042, 13066, 13710, 13835, 14217, 14243, 14267, 14304, 14371, 15139, 15224, 15520, 15643, 15725, 15834, 16599, 16628, 16667, 16990, 17067, 17068, 17098, 17219, 17945, 17965, 18317, 18324, 18355, 18460, 18495, 18537, 19280, 19316, 19340, 19770, 19789, 19808, 19830, 19862, 19867, 20567, 20577, 20590, 20961, 20974, 20983, 21041, 21104, 21127, 21182, 21183, 21918, 21933, 21939, 21980, 22249, 22363, 22373, 22479, 23000, 23198, 23210, 23218, 23248, 23253, 23266, 23583, 23722, 23844, 24550, 24574, 24591, 24615, 24933, 24964, 24990, 25056, 25116, 25845, 25850, 25855, 25871, 25879, 25922, 26239, 26249, 26326, 26327, 26365, 26369, 26415, 27161, 27188, 27208, 27212, 27222, 27298, 27549, 27609, 27614, 27641, 27705, 28487, 28574, 28795, 28905, 28906, 28929, 28982, 29018, 29030, 29043, 29055, 29750, 29887, 29954, 30182, 30278, 31096, 31125, 31128, 31167, 31172, 31176, 31226, 31539, 31546, 31598, 31689, 32267, 32300, 32427, 32538, 33000, 33425, 33614, 33750, 33771, 33808, 33810, 33820, 34110, 34270, 34289, 34329, 34350, 34351, 34362, 34385, 35044, 35142, 35143, 35619, 35642, 35669, 35688, 35714, 36376, 36514, 36522, 36569, 36570, 36648, 37043, 37801, 37854, 38026, 38219, 38221, 38231, 38251, 38260, 38271, 38279, 38311, 38398]
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
