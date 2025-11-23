import argparse
from cmd_args import get_options
from source.src.train import load_GCN_model
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
        for t in test_nodes:
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
        for i, p in enumerate(self.players):
            cf, new_idx, cf_adj, sub_adj, num_nodes, node_dict ,orig_label, cf_label, label, total_loss, loss_pred, loss_graph_dist, found = self.policyQuery(i)
            # self.cf_dict[p.target.item()] = [new_idx, cf_adj, sub_adj, cf, num_nodes, node_dict ,orig_label, cf_label, label, total_loss, loss_pred, loss_graph_dist, found]
            self.cf_dict[p.target] = [new_idx, cf_adj, sub_adj, cf, num_nodes, node_dict, orig_label, cf_label,
                                             label, total_loss, loss_pred, loss_graph_dist, found]
            
        return self.cf_dict
 

if __name__ == "__main__":
    # --dataset cora --use_onehot --use_degree --use_entropy --ent 0.1 --policynet gat --maxbudget 5 --seed 102 --k 4  --verbose --save_prefix inductive_non0_correct_only
    test_model = "GCN"
    dataset_name = "cora"
    gcn_layer = 2
    nhid = 16
    dropout = 0.5
    device = "cpu"
    lr = 0.01
    weight_decay = 0.01
    with_bias = True
    test_nodes = [9, 10, 13, 22, 29, 30, 33, 36, 39, 40, 45, 47, 56, 57, 59, 67, 68, 69, 71, 77, 78, 79, 82, 83, 88, 105, 106, 110, 114, 116, 119, 121, 122, 124, 133, 134, 137, 138, 139, 146, 150, 164, 169, 172, 179, 181, 186, 195, 202, 203, 206, 207, 216, 217, 223, 227, 228, 232, 235, 238, 241, 242, 251, 258, 259, 276, 287, 289, 290, 291, 292, 293, 299, 300, 301, 315, 323, 325, 335, 345, 357, 358, 361, 363, 369, 373, 383, 387, 391, 394, 399, 403, 408, 409, 412, 421, 425, 427, 428, 429, 436, 437, 438, 439, 450, 455, 457, 462, 480, 483, 484, 488, 489, 491, 492, 495, 498, 504, 512, 523, 524, 529, 537, 544, 551, 557, 559, 567, 578, 590, 592, 595, 606, 620, 621, 629, 631, 638, 640, 647, 682, 683, 689, 692, 696, 701, 713, 714, 725, 736, 738, 740, 744, 748, 749, 751, 752, 753, 754, 760, 764, 769, 773, 785, 788, 805, 823, 824, 831, 834, 838, 839, 848, 857, 861, 871, 876, 878, 879, 885, 890, 892, 895, 905, 912, 914, 915, 928, 936, 940, 942, 946, 947, 948, 949, 960, 967, 977, 980, 988, 989, 990, 991, 995, 996, 1004, 1005, 1006, 1008, 1012, 1014, 1015, 1021, 1032, 1039, 1042, 1046, 1049, 1050, 1051, 1058, 1073, 1084, 1085, 1094, 1095, 1098, 1106, 1115, 1117, 1121, 1135, 1138, 1141, 1145, 1153, 1160, 1163, 1165, 1169, 1179, 1185, 1186, 1189, 1191, 1192, 1193, 1196, 1198, 1205, 1207, 1214, 1232, 1236, 1252, 1267, 1275, 1279, 1282, 1290, 1291, 1297, 1299, 1300, 1301, 1303, 1309, 1314, 1316, 1318, 1320, 1326, 1331, 1332, 1333, 1337, 1338, 1340, 1343, 1347, 1352, 1356, 1359, 1365, 1372, 1375, 1376, 1377, 1381, 1386, 1391, 1393, 1397, 1399, 1405, 1407, 1411, 1416, 1421, 1426, 1432, 1433, 1437, 1438, 1439, 1445, 1453, 1455, 1457, 1467, 1472, 1473, 1476, 1477, 1488, 1489, 1490, 1493, 1500, 1504, 1505, 1508, 1509, 1510, 1514, 1515, 1519, 1520, 1532, 1549, 1551, 1554, 1562, 1567, 1569, 1572, 1581, 1582, 1583, 1585, 1586, 1592, 1594, 1595, 1614, 1620, 1623, 1630, 1632, 1638, 1651, 1675, 1678, 1680, 1685, 1688, 1690, 1698, 1702, 1703, 1706, 1715, 1726, 1729, 1732, 1735, 1743, 1746, 1753, 1755, 1759, 1765, 1773, 1779, 1780, 1782, 1783, 1793, 1795, 1796, 1797, 1798, 1802, 1814, 1815, 1816, 1820, 1823, 1827, 1831, 1840, 1847, 1851, 1852, 1865, 1870, 1876, 1877, 1882, 1889, 1891, 1900, 1903, 1905, 1906, 1907, 1912, 1913, 1915, 1925, 1928, 1943, 1972, 1976, 1977, 1983, 1998, 2001, 2004, 2007, 2008, 2009, 2019, 2025, 2032, 2048, 2050, 2051, 2054, 2057, 2058, 2062, 2063, 2064, 2072, 2077, 2079, 2080, 2082, 2099, 2101, 2104, 2124, 2132, 2134, 2141, 2148, 2153, 2155, 2158, 2163, 2177, 2178, 2194, 2204, 2206, 2211, 2218, 2221, 2222, 2232, 2237, 2239, 2240, 2252, 2255, 2258, 2265, 2275, 2284, 2293, 2298, 2299, 2306, 2310, 2321, 2323, 2326, 2336, 2337, 2338, 2346, 2349, 2353, 2366, 2367, 2368, 2369, 2370, 2377, 2380, 2381, 2386, 2387, 2391, 2392, 2397, 2400, 2406, 2407, 2413, 2419, 2432, 2433, 2449, 2468, 2475, 2481]
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
