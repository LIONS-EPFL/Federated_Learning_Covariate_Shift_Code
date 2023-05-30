from functools import partial

import numpy as np
import torch
import torch.optim as optim
from dafl.ratio_estimation import ForcePositive
import wandb

from dafl.args import Args
from dafl.utils import aggregate_gradients, copy_weights
from dafl.logger import getLogger


class Server(object):
    def __init__(self, server_model, clients, criterion, args):
        self.args: Args = args
        self.criterion = criterion
        self.server_model = server_model
        self.clients = clients
        self.logger = getLogger(__name__)

    def loss(self, output, target, reduction="mean"):
        return self.criterion(output, target, reduction=reduction)

    def pred(self, output):
        return output.max(1, keepdim=True)[1]

    def train_ratio_estimators(
        self, 
        combine_testsets=True, 
        on_epoch_end=lambda client, stats: None
    ):
        # Create joint public dataset and estimate ratios
        if combine_testsets:
            joint_public_testset = torch.utils.data.ConcatDataset(
                [client.get_public_testset() for client in self.clients])
        else:
            joint_public_testset = None
        for client in self.clients:
            client_on_epoch_end = partial(on_epoch_end, client)
            client.estimate_ratio(joint_public_testset, on_epoch_end=client_on_epoch_end)

            # TODO: For now force positiveness of ratio estimator
            client.ratio_model = ForcePositive(client.ratio_model)

    def subsample(self, all_client_trainers):
        if self.args.num_subsample == -1:
            return self.clients, all_client_trainers
        else:
            client_idxs = np.random.choice(len(self.clients), self.args.num_subsample, replace=False)
            clients = [self.clients[i] for i in client_idxs]
            client_trainers = [all_client_trainers[i] for i in client_idxs]
            return clients, client_trainers

    def train(self, on_step_end=lambda step: None):
        # Train
        self.server_model.train()
        for client in self.clients:
            client.ratio_model.eval()

        # Set optimizer
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(self.server_model.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(self.server_model.parameters(), 
                lr=self.args.lr, 
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay)

        # Setup
        all_client_trainers = [client.train_iterator() for client in self.clients]

        # Share init
        clients, client_trainers = self.subsample(all_client_trainers)
        self.broadcast_weights(clients)

        for step in range(self.args.num_steps):
            # Compute gradient on each client
            loss = 0.0
            for client_trainer in client_trainers:
                client_loss = next(client_trainer)
                loss += client_loss.item()

            if step % self.args.log_interval == 0:
                self.logger.info(f"Step {step}: loss {loss}")

            # Update server (implicitly updates param.grad)
            aggregate_gradients(self.server_model, 
                                [client.model for client in clients])
            optimizer.step()

            # Update server batch norm with client average
            # self.average_batch_norm_to_server()

            # Subsample for next round and update client weights
            clients, client_trainers = self.subsample(all_client_trainers)
            self.broadcast_weights(clients)

            on_step_end(step)
        self.broadcast_weights(self.clients)
        return step

    def broadcast_weights(self, clients):
        if self.args.batch_norm_agg == "FedBN":
            client_states = [client.model.state_dict() for client in clients]

            for key in self.server_model.state_dict().keys():
                # Copy to clients (excluding batch norm)
                if 'bn' not in key:
                    for state in client_states:
                        state[key].data.copy_(self.server_model.state_dict()[key])

        elif self.args.batch_norm_agg == "FedAvg":
            client_states = [client.model.state_dict() for client in clients]

            # TODO: treat 'num_batches_tracked' differently?
            for key in self.server_model.state_dict().keys():
                # Average batch norm to server
                if 'bn' in key:
                    temp = torch.zeros_like(self.server_model.state_dict()[key], dtype=torch.float32)
                    for state in client_states:
                        temp += state[key]
                    temp = temp / len(clients)
                    self.server_model.state_dict()[key].data.copy_(temp)

                # Copy to clients (including batch norm)
                for state in client_states:
                    state[key].data.copy_(self.server_model.state_dict()[key])
        else:
            raise ValueError   
        # for client in clients:
        #     copy_weights(self.server_model, client.model)

    def average_batch_norm_to_server(self):
        params = zip(*([self.server_model.named_parameters()] + [c.model.named_parameters() for c in self.clients]))
        for param in params:
            server_param_pair = param[0]
            client_param_pairs = param[1:]
            if ".bn" in server_param_pair[0]:
                all = [v.data for k,v in client_param_pairs]
                server_param_pair[1].data = sum(all)/len(all)

    def test(self, clients=None):
        if clients is None:
            clients = self.clients

        self.server_model.eval()
        for client in clients:
            client.model.eval()

        # Test on each testset
        client_test_losses = []
        client_test_accuracies = []
        for i, test_loader in enumerate([client.test_loader for client in clients]):
            test_loss, test_accuracy = self.evaluate_testset(test_loader, clients[i].model)
            client_test_losses.append(test_loss)
            client_test_accuracies.append(test_accuracy)
            self.logger.info(f'Client {i} Test: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
            wandb.log({f'client_{i}_test_acc': test_accuracy, f'client_{i}_test_loss': test_loss}, commit=False)

        # Log max/mean/min of accuracies and losses
        metrics = {'acc': client_test_accuracies, 'loss': client_test_losses}
        for metric_label, metric in metrics.items():
            metric = torch.tensor(metric)
            client_test_metric_min = torch.min(metric)
            client_test_metric_max = torch.max(metric)
            client_test_metric_avg = torch.mean(metric)
            wandb.log({
                f'client_test_{metric_label}_min': client_test_metric_min,
                f'client_test_{metric_label}_max': client_test_metric_max,
                f'client_test_{metric_label}_avg': client_test_metric_avg,
            })
        
        self.server_model.train()
        for client in clients:
            client.model.train()

    def evaluate_testset(self, test_loader, model):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = model(data)
                test_loss += self.loss(output, target).item()
                pred = self.pred(output)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = correct / len(test_loader.dataset)
        return test_loss, test_accuracy
