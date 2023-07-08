import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module
from core import utils
import torch
import numba as nb

class BaseAttack(Module):
    """Abstract base class for target attack classes.
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    device: str
        'cpu' or 'cuda'
    """
    def __init__(self, model, nnodes, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.device = device

        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

        self.modified_adj = None
        self.modified_features = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        Returns
        -------
        None.
        """
        raise NotImplementedError()


class RND(BaseAttack):
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super(RND, self).__init__(model, nnodes, device=device)

    def attack(self, ori_features: sp.csr_matrix, ori_adj: sp.csr_matrix, labels: np.ndarray,
               idx_train: np.ndarray, target_node: int, n_perturbations: int, **kwargs):
        """
        Randomly sample nodes u whose label is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set
        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could be edge removals/additions.
        """
        print(f'number of pertubations: {n_perturbations}')
        modified_adj = ori_adj.tolil()

        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0]
        diff_label_nodes = np.random.permutation(diff_label_nodes)

        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[: n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [x for x in range(ori_adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes, unlabeled_nodes[: n_perturbations-len(diff_label_nodes)]])
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
            pass

        self.modified_adj = modified_adj


# TODO: Implement your own attacker here
class Nettack(BaseAttack):
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super(Nettack, self).__init__(model, nnodes, device=device)
        print(device)
        self.structure_perturbations = []
        # self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []

    def attack(self, ori_features, ori_adj, labels, target_node, n_perturbations, ll_cutoff=0.004, **kwargs):
        ##### Preprocessing #####
        if self.nnodes is None:
            self.nnodes = ori_adj.shape[0]

        self.target_node = target_node

        self.ori_adj = ori_adj.tolil()
        self.modified_adj = ori_adj.tolil()
        self.ori_features = ori_features.tolil()
        # self.modified_features = ori_features.tolil()

        ##### Initialize linear activation function #####
        self.adj_norm = utils.normalize_adj(self.modified_adj)
        self.W = self.get_linearized_weight()
        logits = (self.adj_norm @ self.adj_norm @ self.ori_features @ self.W )[target_node] # (Equation.13)

        ##### Compute sorrogate loss #####
        self.label_u = labels[target_node]
        label_target_onehot = np.eye(int(self.nclass))[labels[target_node]]
        best_wrong_class = (logits - 1000*label_target_onehot).argmax() # axis = 1
        surrogate_losses = [logits[labels[target_node]] - logits[best_wrong_class]] # (Equation.14)

        print("##### Starting attack #####")
        print("##### Performing {} perturbations #####".format(n_perturbations))

        ##### Graph structure preserving pertubations #####
        # Setup starting values of the likelihood ratio test.
        degree_sequence_start = self.ori_adj.sum(0).A1
        current_degree_sequence = self.modified_adj.sum(0).A1
        d_min = 2
        S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
        current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
        n_start = np.sum(degree_sequence_start >= d_min)
        current_n = np.sum(current_degree_sequence >= d_min)
        # Scaling parameter - alpha
        alpha_start = compute_alpha(n_start, S_d_start, d_min)
        # Log-likelihood for the samples (Equation.7)
        log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

        # Initialize influencer nodes
        if len(self.influencer_nodes) == 0:
            # direct attack
            influencers = [target_node]
            self.potential_edges = np.column_stack((np.tile(target_node, self.nnodes-1), np.setdiff1d(np.arange(self.nnodes), target_node)))
            self.influencer_nodes = np.array(influencers)

        self.potential_edges = self.potential_edges.astype("int32")

        for _ in range(n_perturbations):
            print("##### ...{}/{} perturbations ... #####".format(_+1, n_perturbations))

            # Do not consider edges that, if removed, result in singleton edges in the graph.
            singleton_filter = filter_singletons(self.potential_edges, self.modified_adj)
            filtered_edges = self.potential_edges[singleton_filter]

            # Update the values for the power law likelihood ratio test.
            deltas = 2 * (1 - self.modified_adj[tuple(filtered_edges.T)].toarray()[0]) - 1
            
            # A' := A + e
            d_edges_old = current_degree_sequence[filtered_edges]
            d_edges_new = current_degree_sequence[filtered_edges] + deltas[:, None]
            
            # Scalable greedy approximation
            new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
            new_alphas = compute_alpha(new_n, new_S_d, d_min)
            new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
            alphas_combined = compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
            new_ll_combined = compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
            new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)

            # Do not consider edges that, if added/removed, would lead to a violation of the
            # likelihood ration Chi_square cutoff value.
            powerlaw_filter = filter_chisquare(new_ratios, ll_cutoff)
            filtered_edges_final = filtered_edges #[powerlaw_filter]

            ##### Find the log-probabilities that obtains the highest difference #####
            # Compute new entries in A_hat_square_uv
            a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges_final, target_node)
            # Compute the struct scores for each potential edge
            struct_scores = self.struct_score(a_hat_uv_new, self.ori_features @ self.W)
            best_edge_ix = struct_scores.argmin()
            best_edge_score = struct_scores.min()
            best_edge = filtered_edges_final[best_edge_ix]

            # perform edge perturbation
            self.modified_adj[tuple(best_edge)] = self.modified_adj[tuple(best_edge[::-1])] = 1 - self.modified_adj[tuple(best_edge)]
            self.adj_norm = utils.normalize_adj(self.modified_adj)

            self.structure_perturbations.append(tuple(best_edge))
            surrogate_losses.append(best_edge_score)

            # Update likelihood ratio test values
            current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
            current_n = new_n[powerlaw_filter][best_edge_ix]
            current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]

    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores (Equation.15)
        """
        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:,self.label_u]
        struct_scores = logits_for_correct_class - best_wrong_class_logits

        return struct_scores

    def compute_new_a_hat_uv(self, potential_edges, target_node):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.
        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.
        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """
        edges = np.array(self.modified_adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = self.modified_adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, nb.typed.List(edges_set), twohop_ixs, values_before, degrees,
                                         potential_edges.astype(np.int32), target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), self.nnodes])

        return a_hat_uv


    def get_linearized_weight(self):
        surrogate = self.surrogate
        W = surrogate.gc1.weight @ surrogate.gc2.weight
        return W.detach().cpu().numpy()
    
@nb.njit()
def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. (Equation.15)
    """
    N = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < N - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                        degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values

@nb.njit()
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before

def compute_alpha(n, S_d, d_min):
    """
    Approximate the alpha of a power law distribution. (Equation.6)

    """
    return n / (S_d - n * np.log(d_min - 0.5)) + 1 ###

def compute_log_likelihood(n, alpha, S_d, d_min):
    """
    Compute log likelihood of the powerlaw fit. (Equation.7)
    """
    return n * np.log(alpha) + n * alpha * np.log(d_min) - (alpha + 1) * S_d

def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.
    """
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)

    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

    return new_S_d, new_n

def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.
    """
    degs = np.squeeze(np.array(np.sum(adj,0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1

    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0

def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff