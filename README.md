In this project, we aim to discover dynamics (partial differential equations) from in situ videos of scanning transmission electron microscopy.
Mathematically, we want to find the best symbolic equation

<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_t%20%3D%20f(u%2C%20u_x%2C%20u_y%2C%20u_%7Bxx%7D%2C%20u_%7Bxy%7D%2C%20u_%7Byy%7D)">

that can describe the video. <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> is the intensity, a fuction of the x-coordinate, the y-coordinate, and t-coordinate (time). RHS is the temporal derivative, and LHS is an unknown symbolic expression as a function of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au">, the 1st order derivative of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> wrt <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ax">,
<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_x">, the 1st order derivative of <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au"> wrt <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Ay">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_y">, and the three 2nd order derivatives, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Bxx%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Bxy%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0Au_%7Byy%7D">.

### deep learning total variation regularization
![Alt Text](https://media.giphy.com/media/J2V1ppHgClb3RcA3ES/giphy.gif)
![Alt Text](https://media.giphy.com/media/jrnocDRGaJ7VRJnb1w/giphy.gif)
