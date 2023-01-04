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
        normal_distribution<double> distribution(1, 1 / (double) sqrt(prev_sz));
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
    void epoch(datalist data, int batch_size = 100);
    /*Perform one epoch*/
public:
    /*
        Neural Network class to recognize digit images of 28 x 28 pixels.
        By default Three dense layers 784 - 50 - 50 - 10
    */

   DigitNetwork(vector<int> neuron_sizes = {784, 50, 50, 10}, double learning_r = .01) {
     layers = {};
     for (int i = 0; i < neuron_sizes.size() - 1; i++) {
         int from = neuron_sizes[i], to = neuron_sizes[i + 1];
         layers.push_back(Layer(from, to));
     }
     learning_rate = learning_r;
   }


   vector<Matrix<double> > const inline analyze(const Matrix<int> &image);
   /*Calculate neuron activations -- i.e forward propagation, floating point number between 0 - 1 for each neuron */

   void train(const datalist &data, int epochs = 30, int batch_size = 100);
   /*Fit network to given data*/

   double cost_function(const datalist &data);
   /*Evaluatee cost function for the given data. The loss function used is L2*/

};

//inline methods
vector<Matrix<double> > const DigitNetwork::analyze(const Matrix<int> &img){
    vector<double> img_row;
    for(int i = 0;i<img.h;i++){
        for(int j = 0;j<img.w;j++){
            img_row.push_back((double) img.elements[i][j]);
        }
    }
    Matrix<double> input_neurons(img_row);
    input_neurons = input_neurons * (1 / 255); //Normalize
    vector<Matrix<double> > neuron_activations;
    assertm(input_neurons.h == this->layers[0].weights.w, "Network Neurons sizes are mismatching the input size!!!");

    neuron_activations.push_back(input_neurons);
    for(auto lr : this->layers){
        input_neurons = lr.weights * input_neurons + lr.biases;
        for(int i = 0;i<input_neurons.h;i++){
            input_neurons.elements[i][0] = sigmoid(input_neurons.elements[i][0]);
        }
        neuron_activations.push_back(input_neurons);
    }

    return neuron_activations;
}

void back_prop(const int label, vector<Matrix<double> >& grad,const vector<Matrix<double> >& neurons_activation, const vector<Layer>& layers);

#endif