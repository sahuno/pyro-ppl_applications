import torch
import pyro
from pyro.nn import PyroSample
from torch import nn
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from pyro.infer import MCMC, NUTS
import pandas as pd

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features, device="cuda"):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)).expand([out_features, in_features]).to_event(2)) #nfeatures -1
        self.linear.bias = PyroSample(dist.Normal(torch.tensor(0.0, device=device), torch.tensor(10., device=device)).expand([out_features]).to_event(1))
        self.cuda()
    def forward(self, x, y=None, device = "cuda"):
        sigma = pyro.sample("sigma", dist.Uniform(torch.tensor(0., device = device), torch.tensor(10., device = device)))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
#make x,y dataset
def make_data(N, P, mu, sd, device = "cuda"):
    X = dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)).sample((N,P))
    B = dist.Normal(torch.tensor(mu, device=device), torch.tensor(sd, device=device)).sample((P,)) #define betas
    eps = dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)).sample((N,))
    Y = X @ B + eps
    print(f"X.shape= {X.shape}; Y.shape{Y.shape}")
    return X, Y

def train(model, guide, X, Y, lr=0.05, n_steps=201):
    adam_params = {"lr": lr}
    adam = pyro.optim.Adam(adam_params)
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for j in tqdm(range(n_steps)):
        # calculate the loss and take a gradient step
        loss = svi.step(X, Y)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(Y)))

# Utility function to print latent sites' quantile information. #straight from pyro website
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

if __name__ == "__main__":
    print(torch.__version__)
    device = torch.device("cuda")
    X,Y = make_data(N=15, P=3, mu=0.0, sd=10.0)
    # X,Y = make_data(N=300, P=600, mu=0.0, sd=10.0)
    X,Y = X.to("cuda"), Y.to("cuda")
    # #model
    print(f"X.shape[1]; {X.shape[1]}")
    model = BayesianRegression(X.shape[1], 1)
    print(f"\nmodel: {model}\n")

    #visualize model
    #pyro.render_model(model, model_args=(torch.tensor(3.0), torch.tensor(1.0)), filename="model.svg")

    #inference with SVI
    # define guide automatically with inbuilt function. i asusme parameters are idepedent normals
    guide = AutoDiagonalNormal(model)
    print(f"\guide: {guide}\n")
    train(model=model, guide=guide, X=X, Y=Y)
    print(f"get params")
    guide.requires_grad_(False)
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))
    
    print(f"get guide.quantiles([0.25, 0.5, 0.75])")
    print(guide.quantiles([0.25, 0.5, 0.75]))

    #inference using mcmc
    pyro.clear_param_store() #don't interfer with svi posterior
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=20, warmup_steps=200)
    tqdm(mcmc.run(X, Y))
    print(f"done runnign mcmc")
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    print(f"hmc_samples.keys {hmc_samples.keys()}")
    outsahpe = {i: v.shape for i, v in hmc_samples.items()}
    print(f"mcmc samples items {outsahpe}")
    # for site, values in summary(hmc_samples).items():
    #     print("Site: {}".format(site))
    #     print(values, "\n")
    
    print(f"done training")

# to run in unix terminal
# $gpu2
# ~/miniforge3/envs/methyl_ONT/bin/python /home/ahunos/prob_models/pyro_ppl/main_pyro_iris.py
