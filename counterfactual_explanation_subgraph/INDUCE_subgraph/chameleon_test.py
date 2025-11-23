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
    # --dataset chameleon --use_onehot --use_degree --use_entropy --ent 0.1 --policynet gat --maxbudget 5 --seed 102 --k 4  --verbose --save_prefix inductive_non0_correct_only
    test_model = "GCN"
    dataset_name = "chameleon"
    gcn_layer = 3
    nhid = 80
    dropout = 0.55
    device = "cpu"
    lr = 0.001
    weight_decay = 0.001
    with_bias = True
    test_nodes = [11, 29, 33, 40, 43, 51, 57, 76, 82, 90, 91, 113, 125, 138, 143, 178, 180, 182, 187, 194, 201, 208, 220, 221, 223, 235, 237, 239, 253, 258, 263, 265, 272, 296, 311, 314, 324, 346, 347, 369, 370, 375, 378, 379, 380, 389, 406, 425, 491, 534, 541, 576, 584, 589, 590, 606, 622, 623, 640, 664, 667, 671, 687, 694, 719, 720, 729, 746, 751, 758, 760, 761, 778, 782, 786, 801, 827, 879, 889, 900, 915, 920, 941, 948, 956, 973, 999, 1007, 1024, 1040, 1043, 1059, 1072, 1078, 1100, 1105, 1123, 1128, 1144, 1145, 1161, 1188, 1199, 1211, 1221, 1228, 1230, 1231, 1242, 1248, 1266, 1271, 1297, 1308, 1315, 1328, 1332, 1348, 1349, 1351, 1369, 1374, 1382, 1395, 1404, 1406, 1407, 1412, 1413, 1424, 1437, 1439, 1447, 1450, 1461, 1462, 1493, 1508, 1518, 1529, 1548, 1554, 1577, 1580, 1639, 1643, 1659, 1676, 1677, 1684, 1688, 1699, 1706, 1710, 1712, 1719, 1720, 1723, 1742, 1755, 1779, 1792, 1802, 1808, 1818, 1830, 1845, 1859, 1864, 1892, 1901, 1919, 1922, 1940, 1941, 1953, 1962, 1965, 1990, 2001, 2002, 2015, 2022, 2036, 2051, 2071, 2079, 2123, 2141, 2150, 2162, 2172, 2187, 2211, 2236, 2242, 2255, 2261, 2264, 2271]
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
