
def aggregate_gradients(server_model, client_models):
    server_model.zero_grad()
    params = zip(*[server_model.named_parameters()] + [client_model.named_parameters() for client_model in client_models])
    for param in params:
        server_param = param[0]
        client_params = [p[1].grad for p in param[1:] if p[1].grad is not None]

        # Only aggregate if there are client parameters that are not None
        if len(client_params):
            server_param[1].grad = sum(client_params) / len(client_params)


def copy_weights(server_model, client_model):
    params = zip(*[client_model.named_parameters(), server_model.named_parameters()])
    for client_param, server_param in params:
        client_param[1].data = server_param[1].data


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val):
    self.sum += val
    self.count += 1
    self.avg = self.sum / self.count
