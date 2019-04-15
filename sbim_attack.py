import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import LinfPGDAttack,LinfBasicIterativeAttack,L1BasicIterativeAttack,GradientAttack,GradientSignAttack
from layers import SinkhornDistance
from WassNet import WassNet
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='wbim-attack')
parser.add_argument('--seed', default=20, type=int)
parser.add_argument('--beta', default=0, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--iters', default=10, type=int)
parser.add_argument('--eps', default=0.3, type=float)
parser.add_argument('--nb_iter', default=20, type=int)
parser.add_argument('--eps_iter', default=0.02, type=float)
parser.add_argument('--early_stop', default=False, type=bool)
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
filename = "mnist_lenet5_clntrained.pt"

# filename = "mnist_lenet5_advtrained.pt"

# load models
model = LeNet5()
model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location='cpu'))
model.to(device)
model.eval()
wass_model_filename = 'mnist_wass_net.pt'
wass_model = WassNet()
wass_model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, wass_model_filename),map_location='cpu'))
wass_model.to(device)
wass_model.eval()

def myloss(model,yvar,xvar,xadv):
    outputs = model(xadv)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs, yvar)
    if args.beta == 0:
        return loss,Variable(torch.tensor(0.0),requires_grad=True)
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=1000, reduction=None)
    x = xvar.view(-1,28*28,1)
    y= xadv.view(-1,28*28,1)
    w_loss,_,_ = sinkhorn(x, y)
    return loss,-torch.mean(w_loss)


adversary = L1BasicIterativeAttack(
    model, loss_fn=myloss, eps=args.eps,
    nb_iter=args.nb_iter, eps_iter=args.eps_iter, clip_min=0.0, clip_max=1.0,
    targeted=False, beta = args.beta, early_stop = args.early_stop)

def eu_dist(x,y):
    A=x.reshape(-1,28*28)
    B=y.reshape(-1,28*28)
    return np.mean(np.linalg.norm(A-B,ord=2,axis=1))
def s_dist(clns,advs):
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=1000, reduction=None)
    x = np.array(clns).reshape(-1,28*28,1)
    y = np.array(advs).reshape(-1,28*28,1)
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    w_dist,_,_ = sinkhorn(x, y)
    return torch.mean(255*w_dist).numpy()
def w_dist(x,y):
    return (wass_model(x) - wass_model(y)).detach().numpy().mean()

loader = get_mnist_test_loader(batch_size=args.batch_size,shuffle=True)
clns = []
advs = []
trues = []
pred_clns = []
pred_advs = []
iter_counts = []
for batch_idx, (cln_data, true_label) in enumerate(loader):
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    adv_untargeted,iter_count = adversary.perturb(cln_data, true_label)
    iter_counts.append(iter_count)
    pred_cln = predict_from_logits(model(cln_data))
    pred_untargeted_adv = predict_from_logits(model(adv_untargeted))
    clns = clns + list(cln_data.numpy())
    advs = advs + list(adv_untargeted.numpy())
    trues = trues + list(true_label.numpy())
    pred_clns = pred_clns + list(pred_cln.numpy())
    pred_advs = pred_advs + list(pred_untargeted_adv.numpy())
    if (batch_idx == args.iters):
        break

np.save('adv_data/clns.npy',clns)
np.save('adv_data/advs.npy',advs)
np.save('adv_data/true_labels.npy',trues)
np.save('adv_data/pred_clns.npy',pred_clns)
np.save('adv_data/pred_advs.npy',pred_advs)


print('l2 distance:', eu_dist(np.array(clns), np.array(advs)))
print('w distance:', w_dist(torch.tensor(clns).float(), torch.tensor(advs).float()))
print('s distance:', s_dist(torch.tensor(clns).float(), torch.tensor(advs).float()))
print('attack success rate:',1 - (np.array(pred_clns) == np.array(pred_advs)).sum() / len(pred_advs))
print('average iter count:', np.array(iter_counts).mean())
# for ii in range(batch_size):
#     plt.subplot(3, batch_size, ii + 1)
#     _imshow(cln_data[ii])
#     plt.title("clean \n pred: {}".format(pred_cln[ii]))
#     plt.subplot(3, batch_size, ii + 1 + batch_size)
#     _imshow(adv_untargeted[ii])
#     plt.title("untargeted \n adv \n pred: {}".format(
#         pred_untargeted_adv[ii]))
# plt.tight_layout()
# # plt.show()
# plt.savefig("examples.jpg")
