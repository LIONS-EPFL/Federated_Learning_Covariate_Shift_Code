import torch
from torchvision.datasets.mnist import MNIST


from dafl.target_shift import TargetShift, get_targets_counts


def test_target_shift():
    ls = TargetShift(num_classes=3)
    dist = ls.config_to_distribution({0: 0.5, 1: 0.5})
    assert torch.all(dist == torch.tensor([0.5, 0.5, 0.0]))

    ls = TargetShift(num_classes=3)
    dist = ls.config_to_distribution({0: 0.1, 1: 0.5})
    assert torch.all(dist == torch.tensor([0.1, 0.5, 0.4]))

    ls = TargetShift(num_classes=3)
    probs = ls.get_probs([{0: 0.5}, {1: 0.8}], normalize=False)
    assert probs.shape == torch.Size((2, 3))
    assert torch.all(probs == torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]]))

    ls = TargetShift(num_classes=3)
    probs = torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    probs = ls.normalize_probs(probs)
    expected = torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]]) / 1.05
    assert torch.all(probs == expected)
    assert torch.all(torch.sum(probs, 0).le(torch.tensor([1.0000001, 1.0000001, 1.0000001])))

    ls = TargetShift(num_classes=3)
    i = 0
    class_probs = torch.tensor([0.5, 0.1])
    data = torch.arange(10)
    targets = torch.zeros(10)
    data_splits, targets_splits = ls.split_class_data(i, class_probs, data, targets)
    assert len(data_splits) == 2
    assert len(targets_splits) == 2
    assert torch.all(data_splits[0] == torch.tensor([0, 1, 2, 3, 4]))
    assert torch.all(data_splits[1] == torch.tensor([5]))

    ls = TargetShift(num_classes=3)
    i = 0
    class_probs = torch.tensor([0.5, 0.5])
    data = torch.arange(10)
    targets = torch.zeros(10)
    data_splits, targets_splits = ls.split_class_data(i, class_probs, data, targets)
    assert len(data_splits) == 2
    assert len(targets_splits) == 2
    assert torch.all(data_splits[0] == torch.tensor([0, 1, 2, 3, 4]))
    assert torch.all(data_splits[1] == torch.tensor([5, 6, 7, 8, 9]))

    # Maximum class sum 1.05
    ls = TargetShift(num_classes=3)
    data = torch.arange(30)
    targets = torch.cat([torch.zeros(10), torch.ones(10), 2*torch.ones(10)])
    split = [{0: 0.5}, {1: 0.8}]
    splits = ls.split_dataset(data, targets, split, shuffle=False)
    assert len(splits) == 2
    data0 = splits[0].data
    targets0 = splits[0].targets
    data1 = splits[1].data
    targets1 = splits[1].targets
    assert torch.all(targets0 == torch.tensor([0,0,0,0,1,1,2,2]))
    assert torch.all(data0 == torch.cat([torch.arange(4), torch.arange(10, 12), torch.arange(20, 22)]))

    assert torch.all(targets1 == torch.tensor([1,1,1,1,1,1,1]))
    assert torch.all(data1 == torch.cat([torch.arange(12, 12+7)]))

    # Uniform
    ls = TargetShift(num_classes=3)
    data = torch.arange(12)
    targets = torch.cat([torch.zeros(4), torch.ones(4), 2*torch.ones(4)])
    split = [{}, {}]
    splits = ls.split_dataset(data, targets, split, shuffle=False)
    assert len(splits) == 2
    data0 = splits[0].data
    targets0 = splits[0].targets
    data1 = splits[1].data
    targets1 = splits[1].targets
    assert torch.all(targets0 == torch.tensor([0,0,1,1,2,2]))
    assert torch.all(data0 == torch.cat([torch.arange(2), torch.arange(4, 6), torch.arange(8, 10)]))

    assert torch.all(targets1 == torch.tensor([0,0,1,1,2,2]))
    assert torch.all(data1 == torch.cat([torch.arange(2,4), torch.arange(6, 8), torch.arange(10, 12)]))

    # Non-uniform all represented
    ls = TargetShift(num_classes=3)
    data = torch.arange(30)
    targets = torch.cat([torch.zeros(10), torch.ones(10), 2*torch.ones(10)])
    split = [{1: 0.5}, {1: 0.6}] # =[[0.2273, 0.4545, 0.2273], [0.1818, 0.5455, 0.1818]]
    splits = ls.split_dataset(data, targets, split, shuffle=False)
    assert len(splits) == 2
    data0 = splits[0].data
    targets0 = splits[0].targets
    data1 = splits[1].data
    targets1 = splits[1].targets
    assert torch.all(targets0 == torch.tensor([0,0,1,1,1,1,2,2]))
    assert torch.all(data0 == torch.cat([torch.arange(2), torch.arange(10, 14), torch.arange(20, 22)]))

    assert torch.all(targets1 == torch.tensor([0,1,1,1,1,1,2]))
    assert torch.all(data1 == torch.cat([torch.arange(2,3), torch.arange(14, 19), torch.arange(22, 23)]))


def test_make_uniform():
    ls = TargetShift(num_classes=10)
    trainset = MNIST(root='./data', train=False, download=True)
    data, targets = ls.make_uniform(trainset.data, trainset.targets)
    assert data.shape == torch.Size([8920, 1, 28, 28])
    assert {i: 892 for i in range(10)} == get_targets_counts(targets)


def test_get_ratios():
    client_label_dist_test = [{0: 0.5}, {1: 0.5}]
    client_label_dist_train = [{}, {}] # uniform
    target_shift = TargetShift(num_classes=4)
    ratios = target_shift.get_ratios(client_label_dist_test, client_label_dist_train)
    a = 2.0/3.0
    assert torch.all(ratios.isclose(torch.tensor([[2.0, a, a, a],[a, 2.0, a, a]])))


def test_get_ratios_combined():
    client_label_dist_test = [{0: 0.5}, {1: 0.5}]
    client_label_dist_train = [{}, {}] # uniform
    target_shift = TargetShift(num_classes=4)
    ratios = target_shift.get_ratios(client_label_dist_test, client_label_dist_train, combine_testsets=True)
    a = 2.0/3.0
    b = 4.0/3.0
    assert torch.all(ratios.isclose(torch.tensor([[b, b, a, a],[b, b, a, a]])))
