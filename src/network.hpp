#ifndef NETWORK_H
#define NETWORK_H

#include "utils.hpp"

class Layer {
public:
    int size;
    Matrix<float> biases; //biases used to transform to itself
    Matrix<float> weights; //weights used to transform from previous layer to itself

    Layer(int prev_sz, int sz) : 
        size(sz),
        biases(Matrix<float>(vector<float>(sz, .0))),
        weights(Matrix<float>(vector<vector<float> >(sz, vector<float>(prev_sz, 1 / size*prev_sz ))))
    {}
};


class DigitNetwork{
private:
    float learning_rate;
    vector<Layer> layers;
public:
    /*
        Neural Network class to recognize digit images of 28 x 28 pixels.
        Three layers 784 - 250 - 50 - 10
    */

   DigitNetwork(float learning_r = 1.0){
     layers = { Layer(784, 250), Layer(250, 50), Layer(50, 10) };
     learning_rate = learning_r;
   }


   Matrix<float> analyze(Matrix<int> image); //calculate list of certainties, number between 0 - 1 for each digit

   void train(const datalist &data); //update parameters to fit training data

};

#endif