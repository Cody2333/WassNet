import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH

filename = "mnist_lenet5_clntrained.pt"
# filename = "mnist_lenet5_advtrained.pt"

model = LeNet5()
model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location='cpu'))
model.to(device)
model.eval()

batch_size = 5
loader = get_mnist_test_loader(batch_size=batch_size,shuffle=True)
for cln_data, true_label in loader:
    break
cln_data, true_label = cln_data.to(device), true_label.to(device)

from advertorch.attacks import LinfPGDAttack,LinfBasicIterativeAttack,L1BasicIterativeAttack,GradientAttack,GradientSignAttack



adversary = L1BasicIterativeAttack(
    model, eps=0.2,
    nb_iter=10, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
    targeted=False)

adv_untargeted = adversary.perturb(cln_data, true_label)
pred_cln = predict_from_logits(model(cln_data))
pred_untargeted_adv = predict_from_logits(model(adv_untargeted))

for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(adv_untargeted[ii])
    plt.title("untargeted \n adv \n pred: {}".format(
        pred_untargeted_adv[ii]))
plt.tight_layout()
plt.show()
