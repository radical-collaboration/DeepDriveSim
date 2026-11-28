try:
    from . import subgraphs
except ImportError:
    # Running from inside GMVAE/ (e.g. the notebooks)
    import subgraphs  # type: ignore[no-redef]
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.aggregation import MeanMetric

class GMVAE(nn.Module):
    def __init__(self,
                 model_path,
                k=10,
                n_x=784,
                n_z=64,
                qy_dims=[16,16],
                qz_dims=[16,16],
                px_dims=[16,16],
                pz_dims=[16,16],
                r_nent=1,
                batch_size=1000,
                lr=0.0001,
                likelihood="gaussian"):
        super().__init__()

        self.model_path = model_path
        self.k = k  # number of clusters
        self.n_x = n_x # number of input dimension
        self.n_z = n_z  # latent space
        self.qy_dims = qy_dims
        self.qz_dims = qz_dims
        self.px_dims = px_dims
        self.pz_dims = pz_dims
        self.r_nent = r_nent
        self.batch_size = batch_size
        self.lr = lr
        # p(x|z): "gaussian" for continuous inputs, "bernoulli" for binary (contact maps)
        self.likelihood = likelihood

        #Build all layers using subgraphs functions
        self._build_networks()

        #optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=self.lr)

        #Metrics tracking, call it during training loop
        self.train_loss_tracker = MeanMetric()

    def _build_networks(self):
        """
        Build all network layers ONCE using subgraphs layer builders
        """
        # Build q(y|x) network - cluster assignment
        self.qy_layers, self.qy_output_layer = subgraphs.build_qy_layers(
            k=self.k,
            hidden_dims=self.qy_dims
        )

        # Build networks for each cluster
        self.qz_networks = nn.ModuleList()
        self.pz_networks = nn.ModuleList()
        self.px_networks = nn.ModuleList()

        for i in range(self.k):
            # Build q(z|x,y) network for this cluster
            y_transform, qz_hidden, qz_zm, qz_zv = subgraphs.build_qz_layers(
                n_z=self.n_z,
                k=self.k,
                hidden_dims=self.qz_dims
            )
            self.qz_networks.append(nn.ModuleDict({
                'y_transform': y_transform,
                'hidden': qz_hidden,
                'zm': qz_zm,
                'zv': qz_zv
            }))

            # Build p(z|y) netowrk for this cluster
            pz_hidden, pz_zm, pz_zv = subgraphs.build_pz_layers(
                n_z=self.n_z,
                hidden_dims=self.pz_dims
            )
            self.pz_networks.append(nn.ModuleDict({
                'hidden': pz_hidden,
                'zm': pz_zm,
                'zv': pz_zv,
            }))
            # Build p(x|z) network for this cluster
            px_hidden, px_xm, px_xv = subgraphs.build_px_layers(
                n_x=self.n_x,
                hidden_dims=self.px_dims
            )
            self.px_networks.append(nn.ModuleDict({
                'hidden': px_hidden,
                'xm': px_xm,
                'xv': px_xv
            }))

    def forward(self, x, training=False):
        """
        Forward pass through the model using per-built layers.
        """
        # for internal training and evaluation mode to match layer behaviours
        if training:
           self.train()
        else:
           self.eval()

        x = x.to(torch.float32)
        batch_size = x.shape[0]

        #Get cluster assignments using pre-built layers
        qy_logit, qy = subgraphs.execute_qy_graphs(
           x,
           self.qy_layers,
           self.qy_output_layer,
           training=training
           )

        # Process each cluster
        outputs = {
           'qy_logit': qy_logit,
           'qy': qy,
           'z': [],
           'zm': [],
           'zv': [],
           'zm_prior': [],
           'zv_prior': [],
           'xm': [],
           'xv': []
           }

        for i in range(self.k):
            cluster_indices = torch.full((batch_size,),i,dtype=torch.long,device=x.device)
            y_onehot = torch.nn.functional.one_hot(cluster_indices, num_classes=self.k).to(torch.float32)

            # Encode: q(z|x,y) using pre-built layers
            z, zm, zv = subgraphs.execute_qz_graph(
              x, y_onehot,
              self.qz_networks[i]['y_transform'],
              self.qz_networks[i]['hidden'],
              self.qz_networks[i]['zm'],
              self.qz_networks[i]['zv'],
              training=training
             )
            # Prior: p(z|y) using pre-built layers
            _, zm_prior, zv_prior = subgraphs.execute_pz_graph(
               y_onehot,
               self.pz_networks[i]['hidden'],
               self.pz_networks[i]['zm'],
               self.pz_networks[i]['zv'],
               training=training
              )
            # Decoder: p(x|z) using pre-built layers
            xm, xv = subgraphs.execute_px_graph(
               z,
               self.px_networks[i]['hidden'],
               self.px_networks[i]['xm'],
               self.px_networks[i]['xv'],
               training=training
             )

            outputs['z'].append(z)
            outputs['zm'].append(zm)
            outputs['zv'].append(zv)
            outputs['zm_prior'].append(zm_prior)
            outputs['zv_prior'].append(zv_prior)
            outputs['xm'].append(xm)
            outputs['xv'].append(xv)

        return outputs

    def train_steps(self, x):
        x = x.to(torch.float32)

        self.optimizer.zero_grad()

        outputs=self(x,training=True)
        loss, loss_dict = self.compute_loss(x, outputs, training=True)

        loss.backward()

        self.optimizer.step()

        #Update mean metric tracker
        self.train_loss_tracker.update(loss)

        return loss_dict

    def compute_loss(self, x, outputs, training=False):
        """
        Compute ELBO loss using subgraphs.labeled_loss
        """
        x = x.to(torch.float32)
        qy = outputs['qy']

        losses = []
        for i in range(self.k):
            loss_i = subgraphs.labeled_loss(
                self.k,
                x,
                outputs['xm'][i],
                outputs['xv'][i],
                outputs['z'][i],
                outputs['zm'][i],
                outputs['zv'][i],
                outputs['zm_prior'][i],
                outputs['zv_prior'][i],
                likelihood=self.likelihood
             )
            losses.append(loss_i)

        weighted_losses = [qy[:,i] * losses[i] for i in range(self.k)]

        summed_weighted_losses = torch.stack(weighted_losses, dim=0).sum(dim=0)

        nent = -torch.sum(qy * torch.log(qy + 1e-10), dim=-1)

        total_loss = torch.mean(self.r_nent * nent + summed_weighted_losses)

        loss_dict = {
          'loss' : total_loss.item(),
          'nent' : torch.mean(nent).item(),
          'recon' : torch.mean(summed_weighted_losses).item()
          }

        return total_loss, loss_dict
