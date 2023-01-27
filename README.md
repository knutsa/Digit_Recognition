# Digit_Recognition

This is a C++ implementation of a neural network for digit recognition on images from the MNIST database. The highest achieved test accuracy is 95%.
The code implements a simple Neural Network class that uses standard SGD with constant learning rate and mean square error loss, the class could be used for arbitrary classification problems but it has only been tested with the MNIST database.

## How to run
1. Got to `utils.hpp`  and set the `ROOT_FOLDER` macro to the location of the code.
2. Compile all files in the `src` folder and then run the executable, e.g
 ```
  g++ -O3 src/*.cpp -fopenmp -o main
  ./main
 ```
 No libraries needed!
