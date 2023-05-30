"""Based on https://github.com/MasaKat0/D3RE/
"""

import time
from numpy import math

import torch
import torch.optim as optim

from dafl.logger import getLogger


class TrueRatioModel(torch.nn.Module):
    def __init__(self, ratio_target_lookup):
        super().__init__()
        self.ratio_target_lookup = ratio_target_lookup
        self.use_target = True

    def forward(self, data, targets):
        return self.ratio_target_lookup[targets].to(data.device)


class UniformRatioModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor(1.0).repeat(x.shape[0]).to(x.device)


class ForcePositive(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.eps = torch.tensor(0.0001)

    def forward(self, x):
        o = self.model(x)
        #return torch.sign(o)*o
        return torch.max(o, self.eps)


class RatioEstimation(object):
    def __init__(self, id_, ratio_model, trainset, args):
        self.id_ = id_
        self.ratio_model = ratio_model
        self.trainset = trainset
        self.args = args

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.re_batch_size, shuffle=True, **self.args.dataloader_kwargs)

    def train(self, on_epoch_end=None, log=True):
        logger = getLogger(__name__)

        # Set optimizer
        if self.args.re_optimizer == 'adam':
            optimizer = optim.Adam(self.ratio_model.parameters(), 
                lr=self.args.re_lr, 
                weight_decay=self.args.re_weight_decay)
        elif self.args.re_optimizer == 'sgd':
            optimizer = optim.SGD(self.ratio_model.parameters(), 
                lr=self.args.re_lr, 
                momentum=self.args.re_momentum,
                weight_decay=self.args.re_weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=self.args.re_lr_milestones, 
            gamma=0.1)

        # Training
        logger.info('Starting ratio estimation training...')
        self.ratio_model.train()
        start_time = time.time()
        for epoch in range(self.args.re_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for data in self.train_loader:
                inputs, targets, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.args.device), semi_targets.to(self.args.device)
                targets = targets.to(self.args.device)
                
                # Checking data is correct:
                # print(torch.sort(inputs[semi_targets == 0]))
                # print(torch.sort(inputs[semi_targets == 1]))

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                if self.args.model == 'label-based':
                    outputs = torch.t(self.ratio_model(targets))[0]
                else:
                    outputs = torch.t(self.ratio_model(inputs))[0]
                                

                if self.args.re_type == 'lsif':
                    t_nu, t_de = 1-semi_targets, semi_targets
                    n_nu, n_de = max([1., torch.sum(t_nu)]), max([1., torch.sum(t_de)])

                    loss_nu = ((-2*outputs+outputs**2/self.args.re_upper_bound)*t_nu).sum()/n_nu
                    loss_nu_middle = (-outputs**2*t_nu/self.args.re_upper_bound).sum()/n_nu
                    loss_de = (outputs**2*t_de).sum()/n_de

                    if loss_de + loss_nu_middle < 0:
                        loss = - (loss_de + loss_nu_middle)
                    else:
                        loss = loss_nu + loss_nu_middle + loss_de
                elif self.args.re_type == 'pu':
                    positive, unlabeled = semi_targets, 1-semi_targets
                    n_positive, n_unlabeled = max([1., torch.sum(positive)]), max([1., torch.sum(unlabeled)])
                    
                    
                    gp = torch.t(torch.log(1+torch.exp(-outputs)))
                    gu = torch.t(torch.log(1+torch.exp(outputs)))
                
                    loss_positive = (1/self.args.re_upper_bound)*torch.sum(gp*positive)/n_positive
                    loss_negative = torch.sum(gu*unlabeled)/n_unlabeled - (1/self.args.re_upper_bound)*torch.sum(gu*positive)/n_positive
                    
                    if loss_negative < 0:
                        loss = - loss_negative
                    else:
                        loss = loss_positive + loss_negative                

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss = epoch_loss / n_batches

            if log:
                # log epoch statistics
                epoch_train_time = time.time() - epoch_start_time
                logger.info(f'| Epoch: {epoch + 1:03}/{self.args.re_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                            f'| Train Loss: {epoch_loss:.6f} |')
            
            if on_epoch_end is not None:
                on_epoch_end({'epoch': epoch, 'train_loss': epoch_loss, 'lr': scheduler.get_lr()[0]})

            # Step scheduler
            scheduler.step()
            if epoch in self.args.re_lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')
