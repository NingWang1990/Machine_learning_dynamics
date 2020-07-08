In this project, we aim to discover dynamics (partial differential equations) from in situ videos of scanning transmission electron microscopy.
Mathematically, we want to find the best symbolic equation

<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_t%20%3D%20f(u%2C%20u_x%2C%20u_y%2C%20u_%7Bxx%7D%2C%20u_%7Bxy%7D%2C%20u_%7Byy%7D)">

that can describe the video. <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> is the intensity, a fuction of the x-coordinate, the y-coordinate, and t-coordinate (time). RHS is the temporal derivative, and LHS is an unknown symbolic expression as a function of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au">, the 1st order derivative of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> wrt <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ax">,
<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_x">, the 1st order derivative of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> wrt <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ay">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_y">, and the three 2nd order derivatives, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Bxx%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Bxy%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Byy%7D">.

Apparently,we first need to evaluate the numerical derivatives. It is actually a challenging task for experimental data as they are noisy and sparse. The convential approaches, such as the finite difference method, are of little help in this scenario. Interestingly, deep learning offers an elegant way to resolve this challenge. We may parametrize a neural nework for the insitu video, which should be a smooth function of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ax">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ay">, and <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0At">. In order to guarantee the smoothness, I apply the total variation regularization on the neural network, which means that I simply add a regularization term in the loss function,

<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Aloss%20%3D%20R(g)%20%2B%20MSE(g%2Cu)">

where the first term is the regularization term, the second term the mean squared loss, and g the neural network.  

I call this new scheme the [!deep learning total variation regularization]. Once we finish training the neural networks, we can then use the automatic differentiation to obtain numerical derivatives to any order. 

### deep learning total variation regularization
![Alt Text](https://media.giphy.com/media/J2V1ppHgClb3RcA3ES/giphy.gif) 
           ![Alt Text](https://media.giphy.com/media/jrnocDRGaJ7VRJnb1w/giphy.gif)
