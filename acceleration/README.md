# Acceleration algorithms search with unrolled networks

We also investigated two possible approaches for discovery of acceleration: (1) Add **momentum** terms in the unrolled network, which are equivalent to adding the skip connections in the network. (2) Add a **preconditioning** operator in the forward/adjoint steps.

This repository implements unrolled network frameworks for acceleration algorithm search. Our search framework primarily focuses on **momentum acceleration** and **preconditioning** for iterative optimization algorithms. Specifically, we parameterize the momentum terms using historical estimates generated during optimization and model the pseudoinverse preconditioner as an unrolled parameter. Both the momentum terms and preconditioners are trained from data.

## Momentum acceleration
Nesterov's momentum acceleration follows an iterative update scheme given by:

$$
\begin{aligned}
    &\texttt{Step 1. update momentum: } y^k = x^k + \frac{t_{k-1} - 1}{t_k}(x^k - x^{k-1}), \\
    &\texttt{Step 2. update signal: } x^{k+1} = S_{\theta_\eta}\left( y^k - \eta A^T(A y^k - b) \right), \\
    &\texttt{Step 3. update step: } t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}.
\end{aligned}
$$

Nesterov's scheme uses the current estimate $x^k$ and the previous estimate $x^{k-1}$ to form the momentum. Motivated by this idea, we propose a general parameterization of the momentum term and allow the unrolled network to learn the optimal representation from data.

### Learned momentum
Unrolled parameters, $\Theta = \lbrace \theta_i \rbrace$, belong to a general search space and may consist of: (1) gradient step size as one scalar $\alpha$ or as sequence $\lbrace \alpha_k \rbrace _{k=1}^K$ that can remain fixed or change for every iteration. (2) Momentum sequence  $\mathbf{m} = \lbrace \mu_k \rbrace _{k=1}^K$ . (3) Sequence of coefficient $c_i= \lbrace c_k^i \rbrace _{k=1}^K$, to form linear combinations of the history of estimates:  $\mathcal{H}_k(c_i, \lbrace x_i \rbrace)= \sum _{i=0}^{k-1} c_k^i x_i$ , and give a general updating rule:

$$
x_{k+1} = x_k + \mathcal{H}_k(c_i, \lbrace x_i \rbrace) - \alpha\nabla\mathcal{L}(x_k+\mathcal{H}_k(c_i, \lbrace x_i \rbrace)).
$$

By training $\Theta$, we aim to find an optimal way to combine information from previous timesteps, yield an updating rule to solve the target inverse problem effectively and efficiently.


## Pseudo-inverse preconditioner
Existing preconditioning techniques can be broadly categorized into three types: (1) **Right preconditioning**,  where a fixed invertible operator $P$ is applied to the right side of system matrix as $b = APP^{-1}x \equiv APu$, where $u=P^{-1}x$. (2) **Left preconditioning**, where a fixed operator $B$ is applied to the left side of the system matrix and measurements as $Bb=BAx$. (3) **Adaptive preconditioning**, where the gradient direction is modified at every iteration of the the gradient. The adaptive preconditioning is further related to methods like BFGS, adaGrad, RMSProp, and ADAM ([Adaptive preconditioning: AdaGrad and ADAM](https://www.mit.edu/~gfarina/2024/67220s24_L13_adagrad/L13.pdf)).

### Learned preconditioning
In our experiments, we primarily focused on the left preconditioning. We apply a preconditioner $B$ to transform the system as $Bb=BAx$. This transformation does not change the solution but modifies the path taken by the iterative solver. 
The iterative update of the original ISTA algorithm applies adjoint operator on the residual as 

$$
x^{k+1} = \eta_{\theta_k} \left(x^k - \alpha_k A^T(Ax^{k}-b)\right).
$$
The modified algorithm would apply preconditioned operator $A^TU$ as 
$$
x^{k+1} = \eta_{\theta_k} \left(x^k - \alpha_k A^T U (Ax^{k}-b)\right),
$$

where $U = B^TB$. 

We can learn $U$ within our unrolled network or choose $B$ such that $BA$ has better spectral properties than the given $A$. For instance, we can find $B$ by minimizing $\|I-BA\|$, which yields **pseudoinverse** of $A$ as the solution (i.e., $B = A^T(AA^T)^{-1}$ and $U = (AA^T)^{-1}$). This choice of $B$ is also equivalent to the row-orthogonalization of the forward operator, which accelerates the convergence by reducing the total number of iterations.
