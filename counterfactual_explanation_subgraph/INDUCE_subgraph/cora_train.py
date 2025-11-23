import argparse
from ast import arg
import pickle
# import wandb
from utils.dataloader import DataLoader
from torch.utils.data import DataLoader as dl
from torch.nn import DataParallel
from utils.classificationnet import GCNSynthetic
from torch.autograd import Variable
from utils.player import Player
from utils.env import Env
from utils.utils import *
from utils.policynet import *
from cmd_args import get_options
import numpy as np
import time
import torch
import wandb
from deeprobust.graph.defense import GCN, GraphConvolution
from deeprobust.graph.data import Dataset
from torch_geometric.nn import GCNConv


args = get_options()
args.device = 'cpu'
torch.set_printoptions(precision=8)
if(args.device == 'cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" #set according to gpu avalaibility
 
torch.manual_seed(args.seed)
np.random.seed(args.seed)

wandb.init(project=f"InduCE{args.dataset}", config=args)
# wandb.config = {
#   "learning_rate": args.rllr,
#   "budget": args.maxbudget,
#   "episodes": args.maxepisodes,
#   "batch_size": args.batch_size,
#   "beta": args.beta,
#   "discount": args.discount
# }
# wandb.init(config=args)

switcher = {'gcn':PolicyNetwork, 'gat':PolicyNetwork2, 'sage':PolicyNetwork3}
res_file = open('../logs/{}/log_train_{}_{}.txt'.format(args.dataset, args.dataset, args.save_prefix), 'w')
checkpoint_res_file = open('../results/{}/{}_train_checkpoint_{}.pkl'.format(args.dataset, args.dataset, args.save_prefix),'wb')
# PATH = "./saved_models/{}/model_{}_{}.pt".format(args.dataset, args.dataset, args.save_prefix)

#load model to be explained
def loadModel(args, g):
    if(args.saved):
        model = GCNSynthetic(nfeat=g.feats.shape[1], nhid=args.hidden, nout=args.hidden,
                        nclass=len(g.labels.unique()), dropout=args.dropout).to(args.device)
        model.load_state_dict(torch.load(
            "../models/gcn_3layer_{}.pt".format(args.dataset)))
    else: #train model, save it and return
        model = train_model(args, g).to(args.device)

    model.eval()
    output = model(g.feats.to(args.device), g.norm_adj.to(args.device))
    y_pred_orig = torch.argmax(output, dim=1)
    print("y_true counts: {}".format(np.unique(g.labels.cpu().numpy(), return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.cpu().numpy(),
                                                    return_counts=True)))      
    # Confirm model is actually doing something
    train_acc = accuracy(output[g.idx_train], g.labels[g.idx_train])
    test_acc = accuracy(output[g.idx_test], g.labels[g.idx_test])
    print("Train Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)
    print("Train:Test = {} : {}".format(len(g.idx_train),len(g.idx_test)))
    return model

def load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay, with_bias, gcn_layer):
    gnn_model = GCN_initilization(features, labels, nhid, dropout, device, lr, weight_decay, with_bias, gcn_layer)
    gnn_model.load_state_dict(torch.load(file_path))
    gnn_model.eval()
    return gnn_model

def GCN_initilization(features, labels, nhid, dropout, device, lr, weight_decay, with_bias, gcn_layer):
    target_gcn = GCN_extend(nfeat=features.shape[1],
                            nhid=nhid,
                            nclass=labels.max().item() + 1,
                            dropout=dropout,
                            device=device,
                            lr=lr,
                            weight_decay=weight_decay,
                            with_bias=with_bias,
                            gcn_layer=gcn_layer
                            )
    return target_gcn

class GCN_extend(GCN):
    def __init__(self, nfeat, nhid, nclass, dropout, device, lr, weight_decay, with_bias=True, gcn_layer=2):
        super().__init__(nfeat=nfeat,
                         nhid=nhid,
                         nclass=nclass,
                         dropout=dropout,
                         device=device,
                         lr=lr,
                         weight_decay=weight_decay,
                         with_bias=with_bias)
        if gcn_layer == 3:
            self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
            self.gc2 = GraphConvolution(nhid, nhid, with_bias=with_bias)
            self.gc3 = GraphConvolution(nhid, nclass, with_bias=with_bias)
            self.lin = nn.Linear(nhid + nhid + nclass, nclass, bias=with_bias)  # 拼接多层输出
        else:
            self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
            self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.gcn_layer=gcn_layer

    def forward(self, x, adj):
        if self.gcn_layer == 3:
            x1 = F.relu(self.gc1(x, adj))  # 第一层卷积+激活
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = F.relu(self.gc2(x1, adj))  # 第二层卷积+激活
            x2 = F.dropout(x2, self.dropout, training=self.training)
            x3 = self.gc3(x2, adj)  # 第三层无激活
            x = self.lin(torch.cat((x1, x2, x3), dim=1))  # 特征拼接
            return F.log_softmax(x, dim=1)  # 分类输出，输出层使用log_softmax适配NLLLoss
        else:
            x1 = F.relu(self.gc1(x, adj))  # 第一层卷积+激活
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = self.gc2(x1, adj)  # 第二层卷积
            return F.log_softmax(x2, dim=1)  # 分类输出，输出层使用log_softmax适配NLLLoss

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


def log_results(cf_dict):
    final_res = open('../results/{}/{}_train_{}.pkl'.format(args.dataset, args.dataset, args.save_prefix),'wb')
    pickle.dump(cf_dict, final_res)
    final_res.close()


def transfer_weights(dr_model, pyg_model, gcn_layer):
    print("PyG模型结构验证:")
    print(f"conv1.lin存在: {hasattr(pyg_model.conv1, 'lin')}")  # 应为True
    print(f"conv1.lin.weight形状: {pyg_model.conv1.lin.weight.shape}")
    print(f"DeepRobust gc1.weight形状: {dr_model.gc1.weight.shape}")

    # 第一层权重转置 (1433,16) -> (16,1433)
    pyg_model.conv1.lin.weight.data = dr_model.gc1.weight.data.t().clone()
    # pyg_model.conv1.lin.weight.data.copy_(dr_model.gc1.weight.data)
    pyg_model.conv1.bias.data.copy_(dr_model.gc1.bias.data)

    # 第二层权重转置 (16,16) -> (16,16)
    pyg_model.conv2.lin.weight.data = dr_model.gc2.weight.data.t().clone()
    # pyg_model.conv2.lin.weight.data.copy_(dr_model.gc2.weight.data)
    pyg_model.conv2.bias.data.copy_(dr_model.gc2.bias.data)

    if gcn_layer == 3:
        # 第三层权重转置 (16,7) -> (7,16)
        pyg_model.conv3.lin.weight.data = dr_model.gc3.weight.data.t().clone()
        # pyg_model.conv2.lin.weight.data.copy_(dr_model.gc2.weight.data)
        pyg_model.conv3.bias.data.copy_(dr_model.gc3.bias.data)

        # 线性拼接层权重转换
        pyg_model.lin.weight.data.copy_(dr_model.lin.weight.data)
        pyg_model.lin.bias.data.copy_(dr_model.lin.bias.data)
    else:
        pass

    return pyg_model


def GCNtoPYG(gnn_model, device, features, labels, gcn_layer):
    # initialize pyg gcn model
    pyg_gcn = PyGCompatibleGCN(
        in_channels=features.shape[1],
        hidden_channels=nhid,
        out_channels=labels.max().item() + 1,
        dropout=dropout,
        with_bias=with_bias,
        device=device,
        gcn_layer=gcn_layer
    )
    pyg_gcn = pyg_gcn.to(device)

    # Initialize model (using deeprobust adjacency matrix)
    dr_trained_model = gnn_model
    pyg_gcn = transfer_weights(dr_trained_model, pyg_gcn, gcn_layer)
    pyg_gcn.eval()
    return pyg_gcn


class PyGCompatibleGCN(nn.Module):
    """
    2-layer GCN used in GNN Explainer for cora tasks
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout, device, gcn_layer=2, with_bias=True):
        super().__init__()
        self.gcn_layer = gcn_layer
        if self.gcn_layer == 3:
            self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False, bias=with_bias)
            self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False, bias=with_bias)
            self.conv3 = GCNConv(hidden_channels, out_channels, normalize=False, bias=with_bias)
            self.lin = nn.Linear(hidden_channels + hidden_channels + out_channels, out_channels, bias=with_bias)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False, bias=with_bias)
            self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False, bias=with_bias)
        self.dropout = dropout
        self.device = device

    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if self.gcn_layer==3:
            x1 = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = self.conv2(x1, edge_index, edge_weight=edge_weight).relu()
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            x3 = self.conv3(x2, edge_index, edge_weight=edge_weight)
            x = self.lin(torch.cat((x1, x2, x3), dim=1))
            return F.log_softmax(x, dim=1)
        else:
            x1 = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)
            return F.log_softmax(x2, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def node_embeddings(self, sub_x, sub_edge_index):
        # edge_index, _ = add_self_loops(sub_edge_index, num_nodes=sub_x.size(0))
        x = self.conv1(sub_x, sub_edge_index).relu()
        node_embeddings = self.conv2(x, sub_edge_index)
        return node_embeddings

    def graph_embeddings(self, node_embeddings):
        emb = torch.mean(node_embeddings, dim=0)
        # emb = emb / torch.norm(emb)  # Key modification: L2 normalization
        return emb

class SingleTrain(object):
    def __init__(self, args):
        # self.globel_number = 1
        self.args = args
        self.dataset = self.args.dataset
        self.maxbudget = self.args.maxbudget
        self.cf_dict = {}
        
        if(self.dataset not in ['syn1', 'syn4', 'syn5', 'cora']):
            print("Wrong dataset chosen. Terminating ...")
            exit(0)
       
        # loading data
        data_obj = DataLoader(args.dataset, args.add_self_loops) #, args.add_self_loops
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
        
        # training on non zero labels
        if(self.args.train_on_non_zero):
            idx = (self.graph.labels[self.graph.idx_train] > 0).nonzero(as_tuple=False)
            self.targets = self.graph.idx_train[idx]
            if(self.args.verbose):
                print("Number of non zero instances: ", idx.shape[0])
        else:
            self.targets = self.graph.idx_train 
        self.players = []
        self.player_idx = []
        i=0
        for t in self.targets:
            # p = Player(self.graph, t, self.model, args).cuda()
            p = Player(self.graph, t, self.model, args)
            #train only on instances that are correctly classified on extracting k-hop nbrs
            if(self.args.train_on_correct_only): 
                # if(self.graph.labels[t].to(args.device) == p.orig_out):
                #     self.players.append(p)
                #     self.player_idx.append(i)
                #     i+=1
                self.players.append(p)
                self.player_idx.append(i)
                i += 1
            else:
                self.players.append(p)
                self.player_idx.append(i)
                i+=1
            # if(len(self.players) == 4):
            #     break

        if(self.args.verbose):
            print("Number of instances: ", len(self.players))
        self.env = Env(self.players, self.args, torch.max(self.graph.labels)+1)
        #policy distribution depends on num
        # self.policy=switcher[self.args.policynet](self.args,self.env.statedim).cuda() #change - num_actions
        self.policy = switcher[self.args.policynet](self.args, self.env.statedim)  # change - num_actions
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.args.rllr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [300, 600], gamma=0.1, last_epoch=-1)
    
    #add train function here which will call jointtrain for training a batch
    def train(self):
        minloss = 100000000
        training_dataloader = dl(self.player_idx, batch_size=args.batch_size, num_workers=1, shuffle=True)
        for episode in range(0,self.args.maxepisodes): #change made
            cumulative_avg_batch_loss = 0
            if(self.args.verbose):
                print("\n---------- Episode: {} -----------\n".format(episode))
            for idx in training_dataloader:
                batch = []
                batch_ids = []
                for k in idx:
                    k = k.numpy().astype(int).item()
                    batch.append(self.players[k])
                    batch_ids.append(k)
                #batch created
                avg_batch_loss = self.batchtrain(batch, batch_ids)
                cumulative_avg_batch_loss += avg_batch_loss
            if(episode%10 == 0):
                pickle.dump(self.cf_dict, checkpoint_res_file)

            episode_loss = cumulative_avg_batch_loss/len(training_dataloader)
            print("Episode loss: ", episode_loss)
            wandb.log({"Episode loss": episode_loss})

            if(episode_loss < minloss):
                if(self.args.verbose):
                    print("Model updated in episode: ", episode)
                minloss = episode_loss
                torch.save({
                'episode': episode,
                'model_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                }, "../saved_models/{}/model_{}_{}.pt".format(args.dataset, args.dataset, args.save_prefix))
        wandb.watch(self.policy)
        return self.cf_dict    

    def batchtrain(self, batch, batch_ids): #add batch training here     
        # for episode in range(0,self.args.maxepisodes): change made
        rewards_batch, logp_actions_batch, p_actions_batch, entropy_batch = [], [], [], []
        for playerid in batch_ids:
            #cumulate for a batch and backpropagate with that
            rewards, logp_actions, p_actions, entropy = self.playOneEpisode(playerid=playerid)
            rewards_batch.append(rewards)
            logp_actions_batch.append(logp_actions)
            p_actions_batch.append(p_actions)
            entropy_batch.append(entropy)

        p_grad, batch_loss = self.finishEpisode(rewards_batch, logp_actions_batch, entropy_batch)
        avg_batch_loss = batch_loss.mean()
        if(self.args.verbose):
            print("Batch loss (avg): ", avg_batch_loss)  
        return avg_batch_loss   
        
    def playOneEpisode(self, playerid):
        if(self.args.verbose):
            print("\n---------- Player: {} -----------\n".format(playerid))
        self.playerid = playerid
        self.env.reset(playerid)
        rewards, logp_actions, p_actions = [], [], []
        self.states, self.actions = [], []
        entropy_term = 0 
        b = 0 

        orig_out = torch.argmax(self.model(self.players[playerid].G_orig.feats.to(self.args.device), self.players[playerid].G_orig.norm_adj.to(self.args.device))[self.players[playerid].G_orig.target_idx]).item() #self.graph.labels[self.players[playerid].target]#
        curr_out = torch.argmax(self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]).item()
        
        while (orig_out == curr_out and b<self.maxbudget and len(self.players[playerid].cand_dict)>=1):
            state = self.env.getState(playerid) 
            self.states.append(state)
          
            _, logits = self.policy(state[0].to(self.args.device), state[1].to(self.args.device), self.players[playerid].cand_dict)
            action,logp_action, p_action = self.policy.get_action(logits, self.players[playerid].cand_dict)
            entropy = -1*(torch.mean(logits) * torch.log(logits)).sum()
            if(self.args.verbose):
                print("P_action: ", p_action)
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            entropy_term += entropy
            #get all loss values here from player functions
            reward, loss_pred, loss_graph_dist = self.env.step(action,playerid) #passing b as parameter to make beta as beta/sqrt(b) after b>=5
            if(self.args.verbose):
                print("loss total(reward): ", reward)
                print("loss pred: ", loss_pred)
                print("loss graph dist: ", loss_graph_dist)

            rewards.append(reward) 
            curr_out = self.model(self.players[playerid].G_curr.feats.to(self.args.device), self.players[playerid].G_curr.norm_adj.to(self.args.device))[self.players[playerid].G_curr.target_idx]
            curr_out = torch.argmax(curr_out).item()
            b+=1
        
        if(orig_out != curr_out):
            ## dump node_idx, new_idx, "cf_adj", "sub_adj", "y_pred_cf", "y_pred_orig",
            # "label", "num_nodes", "node_dict"
            counterfactual = []
            for i in range(len(self.players[playerid].cf_cand)): #made changes
                val = [self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][0]], self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][1]], self.players[playerid].cf_cand[i][1]]
                counterfactual.append(val)
            
            res_file.write('target: {}, cf: {}\n'.format(self.players[playerid].target.item(), counterfactual)) #self.players[playerid].cf
            self.cf_dict[self.players[playerid].target.item()] = [self.players[playerid].G_curr.target_idx.item(),  self.players[playerid].G_curr.adj, self.players[playerid].G_orig.adj, counterfactual, self.players[playerid].G_curr.adj.shape[0], self.players[playerid].G_curr.node_map, orig_out, curr_out, self.players[playerid].G.labels[self.players[playerid].target], reward, loss_pred, loss_graph_dist, 'found'] #self.players[playerid].cf
        else:
            counterfactual = []
            for i in range(len(self.players[playerid].cf_cand)): #made changes
                val = [self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][0]], self.players[playerid].G_curr.reverse_map[self.players[playerid].cf_cand[i][0][1]], self.players[playerid].cf_cand[i][1]]
                counterfactual.append(val)

            self.cf_dict[self.players[playerid].target.item()] = [self.players[playerid].G_curr.target_idx.item(),  self.players[playerid].G_curr.adj, self.players[playerid].G_orig.adj, counterfactual , self.players[playerid].G_curr.adj.shape[0], self.players[playerid].G_curr.node_map, orig_out, curr_out, self.players[playerid].G.labels[self.players[playerid].target], reward, loss_pred, loss_graph_dist, 'not found'] #self.players[playerid].cf
            res_file.write('target: {}, cf: {}\n'.format(self.players[playerid].target.item(), [])) 
            
        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        rewards = torch.stack(rewards)
    
        return rewards,logp_actions, p_actions, entropy_term
        
    def finishEpisode(self,rewards_batch, log_probs_batch, entropy_batch): #update policy  
        batch_loss = []
        policy_gradient_batch = []
        if(self.args.verbose):
            print("rewards: ", rewards_batch)

        for i in range(len(rewards_batch)):
            rewards = rewards_batch[i]
            log_probs = log_probs_batch[i].squeeze(1)
            entropy = entropy_batch[i]
            
            loss = rewards.sum() #mean/sum, for cumulative rewards of batch which should be minimized
            discounted_rewards = []
            for t in range(len(rewards)):
                Gt = 0 
                pw = 0
                for r in rewards[t:]:
                    Gt = Gt + self.args.discount**pw * (1/r)
                    pw = pw + 1
               
                discounted_rewards = np.append(discounted_rewards,  Gt.cpu().detach().numpy()) #-t or 1/Gt

            #in order to deal with the case when discounted rewards has just one value, so that resulting gradient does not become 0
            if(self.args.verbose):
                    print("discounted rewards single (before whitening): ", discounted_rewards)

            if(len(discounted_rewards) > 1): #.shape[0]
                discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9) # normalize discounted rewards
           
            #check discounted rewards
            vals = torch.tensor(discounted_rewards).to(args.device) 
            actor_loss = (-log_probs * vals).mean()
 
            policy_gradient = actor_loss + self.args.ent*entropy
            policy_gradient_batch.append(policy_gradient)
            batch_loss.append(loss)
            if(self.args.verbose):
                print("entropy single: ", entropy)
                print("policy gradient single: ", policy_gradient)
                print("log probs single: ", log_probs)
                print("discounted rewards single: ", discounted_rewards)
                print("batch loss: ", batch_loss)
                print("Policy gradient batch: ", policy_gradient_batch)

        policy_gradient_batch = torch.tensor(policy_gradient_batch)
        batch_policy_gradient =  Variable(policy_gradient_batch.mean(), requires_grad=True)
        wandb.log({"gradient": batch_policy_gradient})
        if(self.args.verbose):
            print("Batch Policy gradient: ", batch_policy_gradient)
        self.opt.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            batch_policy_gradient.backward()
            self.opt.step()

        return policy_gradient, torch.tensor(batch_loss)


if __name__ == "__main__":
    # --dataset cora --use_onehot --use_degree --use_entropy --ent 0.1 --policynet gat --maxbudget 5 --maxepisodes 10 --seed 102 --train_on_correct_only --k 4 --train_on_non_zero --verbose --save_prefix inductive_non0_correct_only
    test_model = "GCN"
    dataset_name = "cora"
    gcn_layer = 2
    nhid = 16
    dropout=0.5
    device="cpu"
    lr=0.01
    weight_decay=0.01
    with_bias=True

    singletrain = SingleTrain(args)
    start = time.time()
    cf_dict = singletrain.train()
    time_taken = time.time() - start
    print('Time taken (in sec): ', time_taken)
    print('Time taken (in hours): ', time_taken/3600)
    log_results(cf_dict)
    res_file.close()
    checkpoint_res_file.close()
