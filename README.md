In this project, we aim to discover dynamics (partial differential equations) from in situ videos of scanning transmission electron microscopy.
Mathematically, we want to find the best symbolic equation

<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_t%20%3D%20f(u%2C%20u_x%2C%20u_y%2C%20u_%7Bxx%7D%2C%20u_%7Bxy%7D%2C%20u_%7Byy%7D)">

that can describe the video. <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> is the intensity, a fuction of the x-coordinate, the y-coordinate, and t-coordinate (time). RHS is the temporal derivative, and LHS is an unknown symbolic expression as a function of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au">, the 1st order derivative of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> wrt <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ax">,
<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_x">, the 1st order derivative of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> wrt <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ay">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_y">, and the three 2nd order derivatives, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Bxx%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Bxy%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Byy%7D">.

I divide this project into two major steps. The first step is to evaluate numerical derivatives, and the second step is to find the best symbolic equation. 
The challenge in the first step comes from noise and sparsity of experimental data, and the challenge in the second step is to find the global minimum in the symbolic-equation space. To resolve the first challenge, I proposed a scheme, **deep learning total variation regularization**. To resolve the second challenge, I proposed the **spin sequential Monte Carlo** to sample the symbolic-equation space according to the **Bayesian** posterior probability distribution. 

### Deep learning total variation regularization
Apparently, we first need to evaluate numerical derivatives. It is actually a challenging task for experimental data as they are noisy and sparse. The convential approaches, such as the finite difference method, are of little help in this scenario. Interestingly, deep learning offers an elegant way to resolve this challenge. We may parametrize a neural nework for the insitu video, which should be a smooth function of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ax">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ay">, and <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0At">. In order to guarantee the smoothness, I apply the total variation regularization on the neural network, which means that I simply add a regularization term in the loss function,

<img src="https://render.githubusercontent.com/render/math?math=%5Clarge%0Aloss%20%3D%20R(g)%20%2B%20MSE(g%2Cu)">

where the first term is the total-variation regularization term, the second term the mean squared loss, and g the neural network.  
Once we finish training the neural networks, we can then use the automatic differentiation to obtain numerical derivatives to any order. 

Next, I use a simple example to demonstrate this scheme. 
The video below is the soft-segmentation result of a real in situ STEM video,
<p align="center">
<img src="https://media.giphy.com/media/J2V1ppHgClb3RcA3ES/giphy.gif" width="250" height="150" >
</p>.
The signals at the moving interface are very noisy, which prohibits us from using the conventional methods to evaluate the numerical derivatives. Let's use DLTVR to do that.

DLTVR first smoothes the video shown above and return the smoothed video below
<p align="center">
<img src="https://media.giphy.com/media/jrnocDRGaJ7VRJnb1w/giphy.gif" width="250" height="150" >
</p>
We can then employ the automatic differentiation implemeneted in tensorflow or pytorch to calculate the derivatives, which is a piece of cake.


### Spin sequential Monte Carlo
In the second step, I proposed the spin sequential Monte Carlo to find the best partial differential equation (PDE) that can describe the video. I call the algorithm spin sequential Monte Carlo because I combined the sequential Monte Carlo and the spin-flip Markov chain Monte Carlo. The inspiration comes from the paper by [Ruby et al.](https://advances.sciencemag.org/content/3/4/e1602614) and [my PhD work](https://hss-opus.ub.rub.de/opus4/frontdoor/deliver/index/docId/7138/file/diss.pdf). Ruby et al. proposed that the RHS of the partial differential equation might be expressed as a linear combination of non-learn terms

<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0A%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20%3D%20f(u%2C%20u_x%2C%20u_y%2C%20u_%7Bxx%7D%2Cu_%7Bxy%7D%2Cu_%7Byy%7D)%5Cequiv%20%5Calpha_1%20T_1%20%2B%20%20%5Calpha_2%20T_2%20%2B%20...%20%5Calpha_i%20T_i%20%2B%20...%20%5Calpha_N%20T_N%20">

where <img src="https://render.githubusercontent.com/render/math?math=T_1...T_N"> are the non-linear terms in the non-linear library.  

Now the key problem becomes which terms we need to select to construct the PDE. Ruby et al. proposed to use the thresholding Ridge regression, which might work well when the non-linear library is not large but become not suitable for a large non-linear library. [My PhD work](https://hss-opus.ub.rub.de/opus4/frontdoor/deliver/index/docId/7138/file/diss.pdf) on spin systems inpired me to use spin Monte Carlo sampling for this. To be more specific, we may map the linear combination of the non-linear terms onto a *Ising spin chain*. The spin up means that the corresponding term is selected, whereas the spin down means that the corresponding term is not selected. We can then use the spin-flip Markov chain Monte Carlo to sample the PDE space. To speed up sampling efficiency, I combine the spin-flip Monte Carlo with the [sequential Monte Carlo](https://link.springer.com/book/10.1007/978-1-4757-3437-9). 

The probability distribution for a PDE is given as the **Bayesian** posterior

<img src="https://render.githubusercontent.com/render/math?math=%5Clarge%0A%5CP(%5Cmathrm%7BPDE%7D%7C%5Cmathrm%7BData%7D)%20%5Cpropto%20%5Cmathrm%7BLikelihood%7D%20(%5Cmathrm%7BData%7D%7C%20%5Cmathrm%7BPDE%7D)%5Ccdot%20%5Cmathrm%7BPrior%7D(%5Cmathrm%7BPDE%7D)">

which is the product of the likelihood probability and the prior probability. I use the Gaussian error function to define the likelihood probability 

<img src="https://render.githubusercontent.com/render/math?math=%5Clarge%0A%5Cmathrm%7BLikelihood%7D(%5Cmathrm%7BData%7D%20%7C%20%20%5Cmathrm%7BPDE%7D)%20%3D%20%20%5Cfrac%7B1%7D%7B%5Csigma%5Csqrt%7B2%5Cpi%7D%7D%20%5Cmathrm%7Bexp%7D%5Cleft%5B%20%5Cfrac%7B(%5Cmathrm%7Bprediction%7D-%5Cmathrm%7BData%7D)%5E2%7D%7B2%5Csigma%5E2%7D%5Cright%5D,">

and define the prior probability as a function of the PDE complexity

<img src="https://render.githubusercontent.com/render/math?math=%5Clarge%0A%5Cmathrm%7BPrior%7D(%5Cmathrm%7BPDE%7D)%20%3D%20%5Cmathrm%7Bexp%7D%5B-%5Cgamma%20%5Ccdot%20%5Cmathrm%7BComplexity%7D(%5Cmathrm%7BPDE%7D)%5D,">

as we don't want to get too complex PDE. 

We plot the final result for the video shown above as a Pareto frontier for PDE complexity and mean absolute error (MAE):

![alt text](https://github.com/NingWang1990/Machine_learning_dynamics/blob/master/for_readme/Pareto_frontier.png?raw=true)

The best PDE is a first-order hyperbolic PDE, which is not a surprise.
