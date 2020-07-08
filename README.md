In this project, we aim to discover dynamics (partial differential equations) from in situ videos of scanning transmission electron microscopy.
Mathematically, we want to find the best symbolic equation

<img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0A%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20%3D%20f(u%2C%20u_x%2C%20u_y%2C%20u_%7Bxx%7D%2C%20u_%7Bxy%7D%2C%20u_%7Byy%7D)"

that can describe the video. u is the intensity, a fuction of the x-coordinate, the y-coordinate, and t-coordinate (time). RHS is the temporal derivative, and LHS is an unknown symbolic expression as a function of u, the 1st order derivative of u wrt x, u_x, the 1st order derivative of u wrt y, u_y, th three 2nd order derivatives, u_xx, u_xy, u_yy. 
katex.render("a_x", /* element */, {"displayMode":true,"leqno":false,"fleqn":false,"throwOnError":true,"errorColor":"#cc0000","strict":"warn","trust":false,"macros":{"\\f":"f(#1)"}})

### deep learning total variation regularization
![Alt Text](https://media.giphy.com/media/J2V1ppHgClb3RcA3ES/giphy.gif)
![Alt Text](https://media.giphy.com/media/jrnocDRGaJ7VRJnb1w/giphy.gif)
