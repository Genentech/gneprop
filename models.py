import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.nn.functional import linear, logsigmoid, relu, softmax
from torch_geometric.nn import GATConv, global_add_pool, GINEConv, GlobalAttention, JumpingKnowledge, \
    GraphMultisetTransformer, GCNConv
from torch_geometric.nn import global_mean_pool
from math import log


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class GNEpropGIN(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, ffn_hidden_channels, num_layers, out_channels,
                 num_readout_layers, dropout, mol_features_size, aggr='mean', jk='cat', gmt_args=None, use_proj_head=False, proj_dims=(512, 256), skip_last_relu=False):
        super().__init__()

        # graph encoder
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_dim, hidden_channels)

        self.convs = ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1 and skip_last_relu:
                mlp = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(inplace=True),
                    Linear(2 * hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    Dropout(p=dropout)
                )
            else:
                mlp = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(inplace=True),
                    Linear(2 * hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    ReLU(inplace=True),
                    Dropout(p=dropout)
                )
            conv = GINEConv(mlp, train_eps=True)
            self.convs.append(conv)

        self.jk_mode = jk
        if self.jk_mode == 'none':
            self.jk = None
        else:
            self.jk = JumpingKnowledge(mode=self.jk_mode, channels=hidden_channels, num_layers=num_layers)

        # classifier
        self.classifier = ModuleList()

        if self.jk_mode == 'none':
            hidden_channels_mol = hidden_channels + mol_features_size
        elif self.jk_mode == 'cat':
            hidden_channels_mol = hidden_channels * (num_layers + 1) + mol_features_size
        else:
            raise NotImplementedError

        ffn_hidden_size = ffn_hidden_channels if ffn_hidden_channels is not None else hidden_channels_mol

        for layer in range(num_readout_layers):
            input_dim = hidden_channels_mol if layer == 0 else ffn_hidden_size
            mlp = Sequential(
                Linear(input_dim, ffn_hidden_size),
                BatchNorm1d(ffn_hidden_size),
                ReLU(inplace=True),
                Dropout(p=dropout),
            )
            self.classifier.append(mlp)

        # last layer (classifier)
        input_dim = hidden_channels_mol if num_readout_layers == 0 else ffn_hidden_size
        self.classifier.append(Linear(input_dim, out_channels), )

        self.aggr = aggr
        self.global_pool = None
        if aggr == 'mean':
            self.global_pool = global_mean_pool
        elif aggr == 'sum':
            self.global_pool = global_add_pool
        elif aggr == 'global_attention':
            hidden_channels_without_mol = hidden_channels_mol - mol_features_size
            hidden_ga_channels = int(hidden_channels_without_mol / 2)
            gate_nn = Sequential(Linear(hidden_channels_without_mol, hidden_ga_channels),
                                 BatchNorm1d(hidden_ga_channels),
                                 ReLU(inplace=True), Linear(hidden_ga_channels, 1))
            self.global_pool = GlobalAttention(gate_nn)
        elif aggr == 'gmt':
            assert gmt_args is not None

            gmt_sequences = [
                ["GMPool_I"],
                ["GMPool_G"],
                ["GMPool_G", "GMPool_I"],
                ["GMPool_G", "SelfAtt", "GMPool_I"],
                ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"]
            ]

            gmt_sequence = gmt_sequences[gmt_args['gmt_sequence']]

            self.global_pool = GraphMultisetTransformer(in_channels=hidden_channels_mol,
                                                        hidden_channels=gmt_args['hidden_channels'],
                                                        out_channels=hidden_channels_mol,
                                                        Conv=GCNConv,
                                                        num_nodes=200,
                                                        pooling_ratio=gmt_args['gmt_pooling_ratio'],
                                                        pool_sequences=gmt_sequence,
                                                        num_heads=gmt_args['gmt_num_heads'],
                                                        layer_norm=gmt_args['gmt_layer_norm'],)

        self.use_proj_head = use_proj_head
        self.proj_dims = proj_dims
        if self.use_proj_head and self.proj_dims is not None:
            self.proj_head = ModuleList()

            input_dim = hidden_channels_mol
            for proj_dim in self.proj_dims[:-1]:
                mlp = Sequential(
                    Linear(input_dim, proj_dim),
                    BatchNorm1d(proj_dim),
                    ReLU(inplace=True),
                    Dropout(p=dropout),
                )
                self.proj_head.append(mlp)
                input_dim = proj_dim

            # last proj layer
            self.proj_head.append(Linear(input_dim, proj_dims[-1]), )

    def compute_representations(self, x, edge_index, edge_attr, batch, perturb=None):
        list_graph_encodings = []

        x_encoded = self.node_encoder(x)

        edge_attr = self.edge_encoder(edge_attr)

        if perturb is not None:
            if 'perturb_a' in perturb:
                x_encoded += perturb['perturb_a']
            if 'perturb_b' in perturb:
                edge_attr += perturb['perturb_b']

        if self.jk_mode != 'none':
            list_graph_encodings.append(x_encoded)

        for conv in self.convs:
            x_encoded = conv(x_encoded, edge_index, edge_attr)
            if self.jk_mode != 'none':
                list_graph_encodings.append(x_encoded)

        if self.jk_mode != 'none':
            x_encoded = self.jk(list_graph_encodings)
        # x_encoded = torch.stack(list_graph_encodings, dim=1)  # [num_nodes, num_layers, num_channels] # for dnaconv
        # x_encoded = F.relu(self.dna(x_encoded, edge_index)) # for dnaconv

        if self.aggr in ['gmt']:
            out = self.global_pool(x_encoded, batch, edge_index)
        else:
            out = self.global_pool(x_encoded, batch)  # [batch_size, hidden_channels]

        if perturb is not None:
            if 'perturb_graph' in perturb:
                out += perturb['perturb_graph']
        return out

    def forward(self, x, edge_index, edge_attr, mol_features, batch, restrict_output_layers=0, output_type=None,
                perturb=None):
        # compute graph emb
        out_repr = self.compute_representations(x, edge_index, edge_attr, batch, perturb=perturb)
        if mol_features is not None:
            out_repr = torch.cat((out_repr, mol_features), dim=1)

        # compute classifier
        out = out_repr
        for mlp in self.classifier[:None if restrict_output_layers == 0 else restrict_output_layers]:
            out = mlp(out)

        # (optionally) compute proj head
        if self.use_proj_head:
            if self.proj_dims is None:
                out_proj = out_repr[-500:]
            else:
                out_proj = out_repr
                for mlp in self.proj_head:
                    out_proj = mlp(out_proj)

        if output_type is not None:
            if output_type == 'prob':
                out = torch.sigmoid(out)
            elif output_type == 'log_prob':
                out = torch.nn.functional.logsigmoid(out)
            elif output_type == 'prob_multiclass':
                out = torch.sigmoid(out)
                return torch.cat((1 - out, out), dim=-1)
            elif output_type == 'log_prob_multiclass':
                return torch.cat((logsigmoid(-out), logsigmoid(out)),
                                 dim=-1)  # equivalent to log(1-sigmoid(x)), log(sigmoid(x))
            else:
                raise NotImplementedError

        if self.use_proj_head:
            return out, out_proj
        else:
            return out

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mlp in self.classifier:
            mlp.reset_parameters()
        self.input_batch_norm.reset_parameters()


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class SSLEvaluatorLogits(LightningModule):
    def __init__(self, base_model, n_input, n_classes, n_hidden=512, p=0.):
        super().__init__()

        self.base_model = base_model
        self.base_model.freeze()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True),
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x):
        repr = self.base_model(x)
        logits = self.block_forward(repr)
        return logits

    def training_step(self, batch, batch_idx):
        y = batch.y
        y = torch.unsqueeze(y, 1)

        o = self(batch)
        loss = self.loss_function(o, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        y = batch.y
        y = torch.unsqueeze(y, 1)

        o = self(batch)
        loss = self.loss_function(o, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        y = batch.y
        y = torch.unsqueeze(y, 1)

        o = self(batch)
        loss = self.loss_function(o, y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362
        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss