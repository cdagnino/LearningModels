# LearningModels

Provides code for doing value function iteration on models with learning.
The code follows the notation of Aguirregabiria &amp; Jeon (2018): "Firms' Belief and Learning in Oligopoly Markets",
which provides a flexible way of stating a model based on a discrete number of state transition functions.


## Details

$V_{b_t}(I_t)$ is the value of the firm at period $t$ given current information and beliefs.

The value of the firm is given by

$$V_{b_t}(I_t) = max_{a_t \in A} \{ \pi(a_t, x_t) + \beta
                 \int V_{b_{t+1}}(x_{t+1}, I_t) b_t(x_{t+1}| a_t, I_t )\; d x_{t+1}\}  $$

Probably better notation would be to write $V_{b_{t+1}}(I_{t+1}(x_{t+1}, I_t))$

the firm has a prior belief $b_0(x_1 | a_0, x_0)$ that is exogenous.
This prior is a mixture over a collection of L transition probabilities:

$$P = \{p_l (x_{t+1} | a_t, x_t)  \}_{l=1}^L$$

so that

$$b_0(x_1 | a_0, x_0) = \sum_{l=1}^L \lambda_l^{(0)} p_l (x_1| a_0, x_0)$$

The firm observes the new state $x_t$ and uses this information to update its beliefs. This model allows different updating mechanisms, but, if we're using bayesian updating, then it behaves this way:

$$\lambda_l^{(t)} = \frac{ p_l (x_t| a_{t-1}, x_{t-1}) \lambda_l^{(t-1)} }{      \sum_{l'=1}^L p_{l'} (x_t| a_{t-1}, x_{t-1}) \lambda_{l'}^{(t-1)}} $$

In words, $p_l (x_t| a_{t-1}, x_{t-1})$ is the probability that the $l$
transition probability function gives to $x_t$ actually happening. If the probability of $x_t$ (the state that actually occured) is high under $l$, then that $l$ transition probability will get a higher weight in the beliefs of next period.

## A version with demand

In this case the state $x$ from above is the log demand and the action is the price $p$

$$ x = \log q = \alpha + \beta \log p + \varepsilon $$

In the current version, the different $l$ transition probabilities differ in their $\beta_l$. If we assume that $\varepsilon \sim N(0, \sigma^2)$, then


$$p_l(x_{t+1} | p_t, x_{t}) \mbox{  has the pdf of a } N(\alpha + \beta \log p_t, \sigma^2) $$ 

### Expected period profits

The expected period profits are taken over the uncertainty given by the $\lambda$'s.

$$ E[(p-c)* e^{\log q} | I_t] $$

$$ = (p-c)* E[ e^{\log q}| I_t]$$

Let's use $x = \log q$ and write the expected value as

$$ \int e^x b(x) dx = \int e^x \left[\sum_{l=1}^L p_l(x) \lambda_l\right] dx = \sum_{l=1}^L \int e^x  p_l(x) \lambda_l$$

We know that for an $x$ distributed $N(\mu, \sigma^2)$, we have that $E[e^x] = e^{\mu + \sigma^2/2}$, so:

$$\sum_{l=1}^L \int e^x  p_l(x) \lambda_l = \sum_{l=1}^L  e^{\alpha + \beta_l \log p + \sigma^2/2} \lambda_l$$

Therefore

$$E[profit | I_t] = (p-c)* \sum_{l=1}^L  e^{\alpha + \beta_l \log p + \sigma^2/2} \lambda_l$$

$$= (p-c)*e^\alpha*e^{\sigma^2/2}  \sum_{l=1}^L  e^{\beta_l \log p} \lambda_l$$

### Integration

The integration is over future states of log demand.

$$  \int V_{b_{t+1}}(I_{t+1}(x_{t+1}, I_t)) b_t(x_{t+1}| a_t, I_t )\; d x_{t+1} = \int V_{b_{t+1}}(I_{t+1}(x_{t+1}, I_t)) \left[\sum_{l=1}^L p_l(x_{t+1}) \lambda_l\right] \; d x_{t+1} $$


$$ = \sum_{l=1}^L \left[ \int V_{b_{t+1}}(I_{t+1}(x_{t+1}, I_t)) p_l(x_{t+1}) \; d x_{t+1} \right] * \lambda_l $$

Since $p_l(x_{t+1})$ is a normal pdf, it makes sense to use Gauss-Hermite integration for each element of the sum above (See Judd(1998) "Numerical Methods in Economics", page 261ff):

$$\int V_{b_{t+1}}(I_{t+1}(x_{t+1}, I_t)) p_l(x_{t+1}) \; d x_{t+1} \approx \pi^{-1/2} \sum_{i=1}^n w_i V(\sqrt{2} \sigma x_i + \mu_l)$$

where $x_i, w_i$ are Gauss-Hermite collocation points and weights and

$$ \mu_l = \alpha + \beta_l \log p_t $$

According to Judd (1998), ~7 nodes might be enough to get a low error.





### Interpolation

TODO
