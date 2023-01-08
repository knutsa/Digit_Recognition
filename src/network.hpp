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
        weights(Matrix<double>(vector<vector<double> >(sz, vector<double>(prev_sz))))
    {
        default_random_engine generator;
        normal_distribution<double> distribution(0, 2); //This part is probably quite important. Initialize with the right weights and biases!! maybe read up a bit on this
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


   vector<Matrix<double> > inline forward_prop(const Matrix<int> &image);
   /*Calculate neuron activations -- i.e forward propagation, floating point number between 0 - 1 for each neuron */

   Matrix<double> inline analyze(const Matrix<int>& img);

   void train(const datalist &data, int epochs = 30, int batch_size = 100);
   /*Fit network to given data*/

   /*
    Evaluatee cost function and accuracy for the given data. The loss function used is the CrossCategoricalEntropy

    :return cost, accuracy (%)
    */
   pair<double, double> cost_function(const datalist &data);

};

//inline methods
vector<Matrix<double> > inline DigitNetwork::forward_prop(const Matrix<int> &img){
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

Matrix<double> inline softmax(Matrix<double>& output_neurons) {
    Matrix<double> soft_maxed(vector<double>(output_neurons.h, 0));
    double sum = 0;
    for (int i = 0; i < output_neurons.h; i++) {
        auto yi = exp(output_neurons.elements[i][0]);
        soft_maxed.elements[i][0] = yi;
        sum += yi;
    }
    soft_maxed = soft_maxed * (1 / sum);

    return soft_maxed;
}

Matrix<double> DigitNetwork::analyze(const Matrix<int>& img) {
    auto neurons_activation = this->forward_prop(img);
    auto output_neurons = *(neurons_activation.end() - 1);
    //Run logits through softmax

    auto soft_maxed = softmax(output_neurons);
    return soft_maxed;    
}

inline void back_prop(const int label, vector<Matrix<double> >& grad, const vector<Matrix<double> >& neurons_activation, const vector<Layer>& layers, const Matrix<double> output_probabilities) {
    /*Adds contribution of img to grad from img. Grad is a Matrix of the format [weight matrix | biases column]*/
    assert(neurons_activation.size() == grad.size() + 1); //one layer between each group of neurons
    const auto& output_neurons = *(neurons_activation.end() - 1);
    assert(output_probabilities.h == output_neurons.h);
    vector<double> calced(output_probabilities.h, 0); //Derivative with respect to output neurons

    //Derivative of cost function with respect to output neurons -- using cross categorical entropy loss
    double sum = 0, xlabel = output_neurons.elements[label][0];
    for (int i = 0; i < calced.size(); i++) {
        double xi = output_neurons.elements[i][0];
        sum += exp(xi);
    }
    for (int i = 0; i < calced.size(); i++) {
        double xi = output_neurons.elements[i][0];
        calced[i] = exp(xi) / sum;
    }
    calced[label] = exp(xlabel) / sum - 1;

    for (int layer_index = grad.size() - 1; layer_index >= 0; layer_index--) { //process layer of weights in reverse order
        Matrix<double>& mat = grad[layer_index];
        const Layer& current_layer = layers[layer_index];
        assert(mat.h == current_layer.weights.h && mat.w == current_layer.weights.w + 1);
        assert(mat.h == calced.size());
        vector<double> new_calced(mat.w - 1);

        for (int i = 0; i < mat.h; i++) {
            for (int j = 0; j < mat.w; j++) { //last one is the bias
                if (j == mat.w - 1) {
                    mat.elements[i][j] += sigmoid_derivative(calced[i]); //derivative with respect to bias
                }
                else {
                    mat.elements[i][j] += neurons_activation[layer_index].elements[j][0] * sigmoid_derivative(calced[i]); //Derivative with respect to weight
                    new_calced[j] += current_layer.weights.elements[i][j] * sigmoid_derivative(calced[i]); //Derivative with respect to neuron value --- only used to propagate backwards
                }
            }
        }
        calced = new_calced;
    }
}

#endif