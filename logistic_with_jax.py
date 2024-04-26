
#grab from; https://num.pyro.ai/en/stable/handlers.html
#calculates the log likelihoods
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

#create fake data with true distribution
N, D = 3000, 200
data = random.normal(random.PRNGKey(0), (N, D))
true_coefs = jnp.arange(1., D + 1.)
logits = jnp.sum(true_coefs * data, axis=-1)
labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

def logistic_regression(data, labels):
    coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(D), jnp.ones(D))) #set the shrinkage here, 
    intercept = numpyro.sample('intercept', dist.Normal(0., 10.))
    logits = jnp.sum(coefs * data + intercept, axis=-1)
    return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)

#inference with mcmc
num_warmup, num_samples = 1000, 2000
mcmc = MCMC(NUTS(model=logistic_regression), num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(random.PRNGKey(2), data, labels)  
# sample: 100%|██████████| 1000/1000 [00:00<00:00, 1252.39it/s, 1 steps of size 5.83e-01. acc. prob=0.85]
mcmc.print_summary()  


def log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['obs']
    return obs_node['fn'].log_prob(obs_node['value'])

def log_predictive_density(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(lambda rng_key, params: log_likelihood(rng_key, params, model, *args, **kwargs))
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return jnp.sum(logsumexp(log_lk_vals, 0) - jnp.log(n))

print(log_predictive_density(random.PRNGKey(2), mcmc.get_samples(),
      logistic_regression, data, labels))  
# -874.89813