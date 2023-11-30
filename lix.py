import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = np.random.multivariate_normal(mean=[0, 0],
                                  cov=[[1, 0], [0, 1]],
                                  size=300)

y = np.random.multivariate_normal(mean=[0, 0],
                                  cov=[[1, 0.8], [0.8, 1]],
                                  size=300)


class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.init_weights()

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0)

    def load(self):
        pass

    def save(self):
        pass


def mutual_information(joint, marginal, mine_net):
    """
    计算互信息，输出下界
    """
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # 通过引入移动平均(moving average, ma)，可以减小梯度的方差，使得梯度更新更加平滑，从而有助于训练的收敛和稳定性。
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    # use biased estimator
    #     loss = - mi_lb

    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et


def sample_batch(data, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:, 0].reshape(-1, 1),
                                data[marginal_index][:, 1].reshape(-1, 1)],
                               axis=1)
    return batch


def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data, batch_size=batch_size) \
            , sample_batch(data, batch_size=batch_size, sample_mode='marginal')
        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
        if (i + 1) % log_freq == 0:
            print(result[-1])
    return result


def ma(a, window_size=100):
    return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]


mine_net_indep = Mine().cuda()
mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=1e-3)
result_indep = train(x, mine_net_indep, mine_net_optim_indep)

result_indep_ma = ma(result_indep)
print(result_indep_ma[-1])
plt.plot(range(len(result_indep_ma)), result_indep_ma)
plt.show()

mine_net_cor = Mine().cuda()
mine_net_optim_cor = optim.Adam(mine_net_cor.parameters(), lr=1e-3)
result_cor = train(y, mine_net_cor, mine_net_optim_cor)

result_cor_ma = ma(result_cor)
print(result_cor_ma[-1])
plt.plot(range(len(result_cor_ma)), result_cor_ma)
plt.show()

# Test with various correlations
correlations = np.linspace(-0.9, 0.9, 19)
# print(correlations)
final_result = []
for rho in correlations:
    rho_data = np.random.multivariate_normal(mean=[0, 0],
                                             cov=[[1, rho], [rho, 1]],
                                             size=300)
    mine_net = Mine().cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)
    result = train(rho_data, mine_net, mine_net_optim)
    result_ma = ma(result)
    final_result.append(result_ma[-1])
    print(str(rho) + ' : ' + str(final_result[-1]))
    plt.plot(range(len(result_ma)), result_ma)
plt.show()
plt.plot(correlations, final_result)
plt.show()


def func(cov):
    return np.log(np.linalg.det(cov))


def func1(rh, dim=20):
    rh = np.abs(rh)
    dim = dim - 1
    return -0.5 * np.log(dim * rh + 1) - 0.5 * np.log(1 - rh) * dim


