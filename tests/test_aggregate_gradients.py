



import torch
from dafl.utils import aggregate_gradients


DIM = 10

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.weight = torch.nn.parameter.Parameter(torch.randn(DIM))

    def forward(self):
        return self.weight


def test_aggregate_gradients_none():
    server = Model()
    clients = [Model(), Model()]
    aggregate_gradients(server, clients)

    for param in server.parameters():
        assert param.data.grad == None


def test_aggregate_gradients():
    server = Model()
    clients = [Model(), Model()]
    for client_model in clients:
        loss = 1/2 * torch.sum(client_model() ** 2)
        loss.backward()
        
        for param in client_model.parameters():
            assert param.grad is not None
            assert torch.all(torch.isclose(param.grad, param.data))


    aggregate_gradients(server, clients)

    for param in server.parameters():
        print(param.shape)
        expected_grad = sum([next(client_model.parameters()) for client_model in clients]) / 2
        assert torch.all(torch.isclose(param.grad, expected_grad))
