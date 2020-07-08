In this project, we aim to discover dynamics (partial differential equations) from in situ videos of scanning transmission electron microscopy.
Mathematically, we want to find the best symbolic equation

<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_t%20%3D%20f(u%2C%20u_x%2C%20u_y%2C%20u_%7Bxx%7D%2C%20u_%7Bxy%7D%2C%20u_%7Byy%7D)">

that can describe the video the best. <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> is the intensity, a fuction of the x-coordinate, the y-coordinate, and t-coordinate (time). RHS is the temporal derivative, and LHS is an unknown symbolic expression as a function of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au">, the 1st order derivative of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> wrt <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ax">,
<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_x">, the 1st order derivative of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> wrt <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ay">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_y">, and the three 2nd order derivatives, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Bxx%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Bxy%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Byy%7D">.

I divide this project into two major steps. The first step is to evaluate numerical derivatives, and the second step is to find the best symbolic equation. 
The challenge in the first step comes from noise and sparsity of experimental data, and the challenge in the second step is to find the global minimum in the symbolic-equation space. To resolve the first challenge, I proposed a scheme, **deep learning total variation regularization**. To resolve the second challenge, I proposed the **spin sequential Monte Carlo** to sample the symbolic-equation space according to the Bayesian posterior distribution. 

### deep learning total variation regularization
Apparently, we first need to evaluate the numerical derivatives. It is actually a challenging task for experimental data as they are noisy and sparse. The convential approaches, such as the finite difference method, are of little help in this scenario. Interestingly, deep learning offers an elegant way to resolve this challenge. We may parametrize a neural nework for the insitu video, which should be a smooth function of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ax">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ay">, and <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0At">. In order to guarantee the smoothness, I apply the total variation regularization on the neural network, which means that I simply add a regularization term in the loss function,

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
</p>. 
We can then employ the automatic differentiation implemeneted in tensorflow or pytorch to calculate the derivatives, which is a piece of cake.


### Spin sequential Monte Carlo
In the second step, I proposed the spin sequential Monte Carlo to find the best partial differential equation that can describe the video. I call the algorithm spin sequential Monte Carlo because I combined the sequential Monte Carlo and the spin-flip Markov chain Monte Carlo. The inspiration comes from the paper my
