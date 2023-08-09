from utils import *
from models import *

import argparse
import torch
import numpy as np
import random
from sklearn.metrics.cluster import v_measure_score

import warnings
import os
warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def run(args, adj, noisy_adj, node_attributes, node_labels, model_itr, model_path):
    ###################################################
    torch.cuda.empty_cache()
    RANDOM_SEED = model_itr
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    ###################################################

    num_nodes = adj.shape[0]
    num_edges = noisy_adj.sum() / 2
    num_attributes = node_attributes.shape[1]

    node_labels = np.argmax(node_labels, axis=-1)
    num_communites = len(np.unique(node_labels))
    noisy_norm_adj = normalized_adj(noisy_adj)
    aa = adarmic_adar(noisy_adj)

    noisy_adj_th = torch.Tensor(noisy_adj.toarray()).cuda()
    noisy_norm_adj_th = torch.Tensor(noisy_norm_adj.toarray()).cuda()
    node_attributes_th = torch.Tensor(node_attributes.toarray()).cuda()
    aa_th = torch.Tensor(aa).cuda()

    batch_size = args.batch_size
    num_hiddens = args.num_hiddens
    num_epochs = args.num_epochs
    max_epochs = args.max_epochs
    num_pateinces = args.num_pateinces
    c_lr = args.c_lr
    m_lr = args.m_lr
    display_step = args.display_step
    num_batch = int(np.ceil(num_nodes / batch_size))

    c_model_func = globals()["cluster_model"]
    c_model = c_model_func(num_attributes, num_hiddens, num_communites).cuda()

    m_model_func = globals()["meta_model"]
    m_model = m_model_func(num_attributes, num_hiddens).cuda()

    c_optim = torch.optim.Adam(c_model.parameters(),
                               lr=c_lr,
                               weight_decay=0.0)

    m_optim = torch.optim.Adam(m_model.parameters(),
                               lr=m_lr,
                               weight_decay=0.0)

    d = noisy_adj_th.sum(dim=0, keepdim=True).T
    num_edges = noisy_adj_th.sum() / 2
    mod_null = (d @ d.T) / (2 * num_edges)

    best_mod = 0
    for epoch in range(num_epochs):
        c_model.train()
        m_model.train()

        random_node_ids = np.arange(num_nodes)
        np.random.shuffle(random_node_ids)
        cost = 0
        step = "meta_update_1"
        for batch_step in range(num_batch):
            start_id = batch_step * batch_size
            end_id = np.minimum((batch_step + 1) * batch_size, num_nodes)
            if step == "meta_update_1":
                c_model_clone = clone_module(c_model).cuda()
                S = c_model_clone(noisy_norm_adj_th, node_attributes_th) 

                mod_loss = - ((noisy_adj_th - mod_null) * (S @ S.T)) / (2 * num_edges)
                mod_detach = mod_loss.detach()

                weight = m_model(node_attributes_th, noisy_adj_th, aa_th, mod_detach)

                cluster_size = S.sum(dim=0)    
                collapse_loss = torch.norm(cluster_size) / num_nodes * np.sqrt(num_communites) - 1
                
                membership_batch = S[random_node_ids[start_id:end_id], :] @ S.T
                weight_batch = weight[random_node_ids[start_id:end_id], :]
                difference = noisy_adj_th[random_node_ids[start_id:end_id], :] - mod_null[random_node_ids[start_id:end_id], :]
                loss_with_meta = - ((weight_batch * difference) * membership_batch).sum() / (2 * num_edges) + ((end_id - start_id) / num_nodes) * collapse_loss

                grad = torch.autograd.grad(loss_with_meta,  c_model_clone.parameters(), create_graph=True)

                updates = [-c_lr * g for g in grad]
                c_model_clone = update_module(c_model_clone, updates=updates)

                prev_start_id = start_id
                prev_end_id = end_id
                step = "meta_update_2"
                continue

            elif step == "meta_update_2":
                S = c_model_clone(noisy_norm_adj_th, node_attributes_th)

                membership_batch = S[random_node_ids[start_id:end_id], :] @ S.T
                difference = noisy_adj_th[random_node_ids[start_id:end_id], :] - mod_null[random_node_ids[start_id:end_id], :]
                mod_loss = - (difference * membership_batch).sum() / (2 * num_edges) 
                m_optim.zero_grad()
                mod_loss.backward()
                m_optim.step()
                del c_model_clone

                weight = m_model(node_attributes_th, noisy_adj_th, aa_th, mod_detach)
                S = c_model(noisy_norm_adj_th, node_attributes_th)
                cluster_size = S.sum(dim=0)    
                collapse_loss = torch.norm(cluster_size) / num_nodes * np.sqrt(num_communites) - 1
                
                membership_batch = S[random_node_ids[prev_start_id:prev_end_id], :] @ S.T
                weight_batch = weight[random_node_ids[prev_start_id:prev_end_id], :]
                difference = noisy_adj_th[random_node_ids[prev_start_id:prev_end_id], :] - mod_null[random_node_ids[prev_start_id:prev_end_id], :]
                loss_with_meta = - ((weight_batch * difference) * membership_batch).sum() / (2 * num_edges) + ((prev_end_id - prev_start_id) / num_nodes) * collapse_loss

                c_optim.zero_grad()
                loss_with_meta.backward()
                c_optim.step()
                step = "meta_update_1"
                cost += loss_with_meta

        if (epoch + 1) % display_step == 0:
            c_model.eval()
            m_model.eval()
            S = c_model(noisy_norm_adj_th, node_attributes_th)
            S_np = S.detach().cpu().numpy()
            predict = np.argmax(S_np, axis=-1)

            mod = modularity(adj, predict)
            nmi = v_measure_score(node_labels, predict)
            precision = pairwise_precision(node_labels, predict)
            recall = pairwise_recall(node_labels, predict)
            F1 = 2 * precision * recall / (precision + recall)
            print("Epoch %d:%d  loss: %.2f  modularity: %.3f  NMI: %.3f  F1 Score: %.3f" % (epoch+1, num_epochs, cost, mod, nmi, F1))
            
            torch.save({
                "epoch": epoch,
                "m_model_state_dict": m_model.state_dict(),
                "c_model_state_dict": c_model.state_dict(),
                "num_nodes": num_nodes,
                "num_attributes": num_attributes,
                "num_hiddens": num_hiddens,
                "num_communites": num_communites,
                "m_optim_state_dict": m_optim.state_dict(),
                "c_optim_state_dict": c_optim.state_dict(),
                "batch_size": batch_size,
                "c_lr": c_lr,
                "m_lr": m_lr,
            }, "model.tar")
            if mod > best_mod: best_mod = mod

    epoch += 1
    num_pateinces_count = 0
    last_log = ""
    while True:
        c_model.train()
        m_model.train()

        random_node_ids = np.arange(num_nodes)
        np.random.shuffle(random_node_ids)

        step = "meta_update_1"
        for batch_step in range(num_batch):
            start_id = batch_step * batch_size
            end_id = np.minimum((batch_step + 1) * batch_size, num_nodes)
            if step == "meta_update_1":
                c_model_clone = clone_module(c_model).cuda()
                S = c_model_clone(noisy_norm_adj_th, node_attributes_th) 

                mod_loss = - ((noisy_adj_th - mod_null) * (S @ S.T)) / (2 * num_edges)
                mod_detach = mod_loss.detach()
                weight = m_model(node_attributes_th, noisy_adj_th, aa_th, mod_detach)
                cluster_size = S.sum(dim=0)    
                collapse_loss = torch.norm(cluster_size) / num_nodes * np.sqrt(num_communites) - 1
                
                membership_batch = S[random_node_ids[start_id:end_id], :] @ S.T
                weight_batch = weight[random_node_ids[start_id:end_id], :]
                difference = noisy_adj_th[random_node_ids[start_id:end_id], :] - mod_null[random_node_ids[start_id:end_id], :]
                loss_with_meta = - ((weight_batch * difference) * membership_batch).sum() / (2 * num_edges) + ((end_id - start_id) / num_nodes) * collapse_loss
                grad = torch.autograd.grad(loss_with_meta, c_model_clone.parameters(), create_graph=True)

                updates = [-c_lr * g for g in grad]
                c_model_clone = update_module(c_model_clone, updates=updates)

                prev_start_id = start_id
                prev_end_id = end_id
                step = "meta_update_2"
                continue

            elif step == "meta_update_2":
                S = c_model_clone(noisy_norm_adj_th, node_attributes_th)

                membership_batch = S[random_node_ids[start_id:end_id], :] @ S.T
                difference = noisy_adj_th[random_node_ids[start_id:end_id], :] - mod_null[random_node_ids[start_id:end_id], :]
                mod_loss = - (difference * membership_batch).sum() / (2 * num_edges) 
                m_optim.zero_grad()
                mod_loss.backward()
                m_optim.step()
                del c_model_clone

                weight = m_model(node_attributes_th, noisy_adj_th, aa_th, mod_detach)
                S = c_model(noisy_norm_adj_th, node_attributes_th)
                cluster_size = S.sum(dim=0)    
                collapse_loss = torch.norm(cluster_size) / num_nodes * np.sqrt(num_communites) - 1
                
                membership_batch = S[random_node_ids[prev_start_id:prev_end_id], :] @ S.T
                weight_batch = weight[random_node_ids[prev_start_id:prev_end_id], :]
                difference = noisy_adj_th[random_node_ids[prev_start_id:prev_end_id], :] - mod_null[random_node_ids[prev_start_id:prev_end_id], :]
                loss_with_meta = - ((weight_batch * difference) * membership_batch).sum() / (2 * num_edges) + ((prev_end_id - prev_start_id) / num_nodes) * collapse_loss

                c_optim.zero_grad()
                loss_with_meta.backward()
                c_optim.step()
                step = "meta_update_1"
                cost += loss_with_meta

        c_model.eval()
        m_model.eval()
        S = c_model(noisy_norm_adj_th, node_attributes_th)
        S_np = S.detach().cpu().numpy()
        predict = np.argmax(S_np, axis=-1)
        mod = modularity(adj, predict)

        if best_mod < mod:
            mod = modularity(adj, predict)
            nmi = v_measure_score(node_labels, predict)
            precision = pairwise_precision(node_labels, predict)
            recall = pairwise_recall(node_labels, predict)
            F1 = 2 * precision * recall / (precision + recall)
            last_log = "Epoch %d:%d  loss: %.2f  modularity: %.3f  NMI: %.3f  F1 Score: %.3f" % (epoch+1, num_epochs, cost, mod, nmi, F1)
            torch.save({
                "epoch": epoch,
                "m_model_state_dict": m_model.state_dict(),
                "c_model_state_dict": c_model.state_dict(),
                "num_nodes": num_nodes,
                "num_attributes": num_attributes,
                "num_hiddens": num_hiddens,
                "num_communites": num_communites,
                "m_optim_state_dict": m_optim.state_dict(),
                "c_optim_state_dict": c_optim.state_dict(),
                "batch_size": batch_size,
                "c_lr": c_lr,
                "m_lr": m_lr,
            }, "model.tar")
            best_mod = mod
            num_pateinces_count = 0 
        else:
            num_pateinces_count += 1
        epoch += 1
        
        if num_pateinces_count == num_pateinces or epoch == max_epochs:
            if not last_log == "": print(last_log)
            break

    checkpoint = torch.load("model.tar")
    c_model_state_dict = checkpoint["c_model_state_dict"]
    c_model.load_state_dict(c_model_state_dict)
    c_model.eval()
    S = c_model(noisy_norm_adj_th, node_attributes_th)
    S_np = S.detach().cpu().numpy()

    predict = np.argmax(S_np, axis=-1)
    mod = modularity(adj, predict)
    nmi = v_measure_score(node_labels, predict)
    precision = pairwise_precision(node_labels, predict)
    recall = pairwise_recall(node_labels, predict)
    F1 = 2 * precision * recall / (precision + recall)
    return [mod, nmi, F1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source code of Robust Graph Clustering via Meta Learning for Noisy Graphs")
    parser.add_argument("--graph_name", type=str, default="cora")

    parser.add_argument("--noise_level", type=int, default=1, help="noise level, [1, 2, 3] (1 --> 30%, 2 --> 60%, 3 --> 90%)")
    parser.add_argument("--batch_size", type=int, default=128, help="set the batch sizes (i.e., the numbers of nodes in a batch)")
    parser.add_argument("--num_hiddens", type=int, default=64, help="number of hidden units in the GCN variant")
    parser.add_argument("--num_epochs", type=int, default=200, help="minimum epochs")
    parser.add_argument("--max_epochs", type=int, default=1500, help="maximum epochs")
    parser.add_argument("--num_pateinces", type=int, default=50, help="number of epochs with no improvement of modulairty after which training will be stopped")
    parser.add_argument("--c_lr", type=float, default=0.001, help="learning rate of the clustering model")
    parser.add_argument("--m_lr", type=float, default=0.005, help="learning rate of the meta model")

    parser.add_argument("--display_step", type=int, default=100, help="Display step for monitoring training")
    parser.add_argument("--num_noise_itr", type=int, default=5, help="Number of noise graphs for generate")
    parser.add_argument("--num_model_itr", type=int, default=3, help="Number of iteration to training model for averaging")
    args = parser.parse_args()

    adj, node_attributes, node_labels = load_graph(args.graph_name)

    avg_mod, avg_nmi, avg_f1 = 0, 0, 0
    num_itr = args.num_noise_itr * args.num_model_itr
    model_path = args.graph_name + "_%d.tar"%(args.noise_level)
    for noise_itr in range(args.num_noise_itr):
        noisy_adj = add_noise_edges(adj=adj, 
                                    node_labels=node_labels, 
                                    noise_level=args.noise_level, 
                                    random_seed=noise_itr)
        for model_itr in range(args.num_model_itr):
            mod, nmi, f1 = run(args, adj, noisy_adj, node_attributes, node_labels, model_itr, model_path)
            itr = args.num_model_itr * noise_itr + model_itr
            print("  ITER: %d/%d   MOD: %.3f  NMI: %.3f  F1: %.3f" % (itr+1, num_itr, mod, nmi, f1))
            avg_mod += mod
            avg_nmi += nmi
            avg_f1 += f1
    avg_mod /= num_itr
    avg_nmi /= num_itr
    avg_f1 /= num_itr
    print("-----------------------------------------------")
    print("[FINAL]  MOD: %.3f  NMI: %.3f  F1: %.3f" % (avg_mod, avg_nmi, avg_f1))