import os
import numpy as np
import torch
from counterfactual_explanation_subgraph.CFF.explainer_models import NodeExplainerEdgeMulti
from counterfactual_explanation_subgraph.CFF.gcn import GCNNodeBAShapes
from utilty.ba_shapes_preprocessing import BAShapesDataset
import sys
import argparse

res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(os.path.dirname(res))
sys.path.insert(0, base_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="BA_Shapes",
                        help="choose a node explanation task")
    parser.add_argument("--model_path", dest="model_path", type=str, default="log/BA_Shapes_logs",
                        help="path to the model that need to be explained")
    parser.add_argument("--gpu", dest="gpu", action="store_true", default=False, help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='0.005',
                        help="L2 norm to the wights")
    parser.add_argument("--opt", dest="opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", dest="lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=500, help="number of the training epochs")
    parser.add_argument("--lam", dest="lam", type=float, default=500,
                        help="hyper param control the trade-off between "
                             "the explanation complexity and explanation strength")
    parser.add_argument("--alp", dest="alp", type=float, default=0,
                        help="hyper param control factual and counterfactual")
    parser.add_argument("--gam", dest="gam", type=float, default=0.5, help="margin value for bpr loss")
    parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=.5,
                        help="threshold to convert relaxed adj matrix to binary")
    return parser.parse_args()


if __name__ == "__main__":
    SEED_NUM = 102

    torch.manual_seed(SEED_NUM)
    np.random.seed(SEED_NUM)

    np.set_printoptions(threshold=sys.maxsize)
    exp_args = get_args()
    print("argument:\n", exp_args)
    model_path = base_path + '/' + exp_args.model_path
    train_indices = np.load(os.path.join(model_path, 'train_indices.pickle'), allow_pickle=True)
    test_indices = np.load(os.path.join(model_path, 'test_indices.pickle'), allow_pickle=True)
    G_dataset = BAShapesDataset(load_path=os.path.join(model_path))
    # targets = np.load(os.path.join(model_path, 'targets.pickle'), allow_pickle=True)  # the target node to explain
    graphs = G_dataset.graphs
    labels = G_dataset.labels
    targets = G_dataset.targets
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    base_model = GCNNodeBAShapes(G_dataset.feat_dim, 100, num_classes=4, device=device, if_exp=True).to(device)
    base_model.load_state_dict(torch.load(os.path.join(model_path, 'model.model')))
    #  fix the base model
    for param in base_model.parameters():
        param.requires_grad = False
    # Create explainer
    explainer = NodeExplainerEdgeMulti(
        base_model=base_model,
        G_dataset=G_dataset,
        args=exp_args,
        test_indices=test_indices,
        # fix_exp=6
    )

    explainer.explain_nodes_gnn_stats()
