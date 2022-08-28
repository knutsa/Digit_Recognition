#ifndef NETWORK_H
#define NETWORK_H

#include "utils.hpp"

#define sigmoid(x) 1 / (1 + exp(-x))
#define sigmoid_derivative(x) 1 / ( (exp(x/2) + exp(-x/2)) *(exp(x/2)+exp(-x/2)) )

class Layer {
public:
    int size;
    Matrix<double> biases; //biases used to transform to itself
    Matrix<double> weights; //weights used to transform from previous layer to itself

    Layer(int prev_sz, int sz) : 
        size(sz),
        biases(Matrix<double>(vector<double>(sz, .0))),
        weights(Matrix<double>(vector<vector<double> >(sz, vector<double>(prev_sz, 1 / size*prev_sz ))))
    {
        default_random_engine generator;
        normal_distribution<double> distribution(.0, 1 / (double) sqrt(prev_sz));
        for(int i = 0;i<weights.h;i++){
            for(int j = 0;j<weights.w;j++){
                weights.elements[i][j] =  distribution(generator);
            }
        }
    }
};


class DigitNetwork{
private:
    double learning_rate;
    vector<Layer> layers;
public:
    /*
        Neural Network class to recognize digit images of 28 x 28 pixels.
        Three layers 784 - 250 - 50 - 10
    */

   DigitNetwork(double learning_r = 1.0){
     layers = { Layer(784, 250), Layer(250, 50), Layer(50, 10) };
     learning_rate = learning_r;
   }


   vector<Matrix<double> > const inline analyze(const Matrix<int> &image); //calculate neuron activations -- i.e forward propagation, number between 0 - 1 for each neuron

   void train(const datalist &data); //update parameters to fit training data

   double cost_function(const datalist &data); // determine cost value with current parameters

};

//inline methods
vector<Matrix<double> > const DigitNetwork::analyze(const Matrix<int> &img){
    vector<double> img_row;
    for(int i = 0;i<img.h;i++){
        for(int j = 0;j<img.w;j++){
            img_row.push_back((double) img.elements[i][j]);
        }
    }
    Matrix<double> data(img_row);
    vector<Matrix<double> > neuron_activations;
    neuron_activations.push_back(data);
    for(auto lr : this->layers){
        data = lr.weights * data + lr.biases;
        for(int i = 0;i<data.h;i++){
            data.elements[i][0] = sigmoid(data.elements[i][0]);
        }
        neuron_activations.push_back(data);
    }

    return neuron_activations;
}

#endif