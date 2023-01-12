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
    void epoch(datalist data, int batch_size = 100);
    /*Perform one epoch*/
public:
    /*
        Neural Network class to recognize digit images of 28 x 28 pixels.
        By default Three dense layers 784 - 50 - 50 - 10
    */

    vector<Layer> layers;

   DigitNetwork(vector<int> neuron_sizes = {784, 50, 50, 10}, double learning_r = .01) {
     layers = {};
     for (int i = 0; i < neuron_sizes.size() - 1; i++) {
         int from = neuron_sizes[i], to = neuron_sizes[i + 1];
         layers.push_back(Layer(from, to));
     }
     learning_rate = learning_r;
   }

   /*Calculate neuron activations -- i.e forward propagation, floating point number between 0 - 1 for each neuron */
   vector<vector<double> > inline forward_prop(const Matrix<int> &image);

   vector<double> inline analyze(const Matrix<int>& img);

   void train(const datalist &data, int epochs = 30, int batch_size = 100);
   /*Fit network to given data*/

   /*
    Evaluatee cost function and accuracy for the given data. The loss function used is the CrossCategoricalEntropy

    :return cost, accuracy (%)
    */
   pair<double, double> cost_function(const datalist &data);

};

//inline methods
vector<vector<double> > inline DigitNetwork::forward_prop(const Matrix<int> &img){
    vector<double> img_row;
    for(int i = 0;i<img.h;i++){
        for(int j = 0;j<img.w;j++){
            img_row.push_back((double) img.elements[i][j]);
        }
    }
    Matrix<double> input_neurons(img_row);
    input_neurons = input_neurons * (1 / 255.0); //Normalize
    vector<vector<double> > neuron_activations;
    assertm(input_neurons.h == this->layers[0].weights.w, "Network Neurons sizes are mismatching the input size!!!");

    neuron_activations.push_back(input_neurons.transpose().elements[0]);
    for(auto lr : this->layers){
        input_neurons = lr.weights * input_neurons + lr.biases;
        vector<double> neuron_activation(input_neurons.h);
        for(int i = 0;i<input_neurons.h;i++){
            double yi = sigmoid(input_neurons(i));
            neuron_activation[i] = input_neurons.elements[i][0] = yi;
        }
        neuron_activations.push_back(neuron_activation);
    }
    return neuron_activations;
}

vector<double> inline softmax(const vector<double>& output_neurons) {
    vector<double> soft_maxed(output_neurons.size(), 0);
    double sum = 0.0;
    for (int i = 0; i < output_neurons.size(); i++) {
        auto yi = exp(output_neurons[i]);
        soft_maxed[i] = yi;
        sum += yi;
    }
    for (int i = 0; i < soft_maxed.size(); i++){ soft_maxed[i] /= sum; }

    return soft_maxed;
}

vector<double> DigitNetwork::analyze(const Matrix<int>& img) {
    auto neurons_activation = this->forward_prop(img);
    auto &output_neurons = *(neurons_activation.end() - 1);
    //Run logits through softmax

    auto soft_maxed = softmax(output_neurons);
    return soft_maxed;    
}

/*Add to_add to the doubles stored at p (simd version of +=)*/
inline void vecadd_pd(double* p, __m256d to_add) {
    __m256d loaded = _mm256_loadu_pd(p);
    __m256d updated = _mm256_add_pd(loaded, to_add);
    _mm256_storeu_pd(p, updated);
}

//This function is the bottleneck of the algorithm, it has been modified to run a little faster
/*Adds contribution of img to grad from img. Grad is a Matrix of the format [weight matrix | biases column], minimizing use of matrices to try and reduce computation time.*/
inline void back_prop(const int label, vector<vector<vector<double> > >& grad, const vector<vector<double> >& neurons_activation, const vector<Layer>& layers, const vector<double> output_probabilities) {
    assert(neurons_activation.size() == grad.size() + 1); //one layer between each group of neurons
    const auto& output_neurons = *(neurons_activation.end() - 1);
    assert(output_probabilities.size() == output_neurons.size());
    vector<double> calced(output_probabilities.size(), 0); //Derivative with respect to output neurons

    //Derivative of cost function with respect to output neurons -- using cross categorical entropy loss
    double sum = 0, xlabel = output_neurons[label];
    for (int i = 0; i < calced.size(); i++) {
        double xi = output_neurons[i];
        sum += exp(xi);
    }
    for (int i = 0; i < calced.size(); i++) {
        double xi = output_neurons[i];
        calced[i] = exp(xi) / sum;
    }
    calced[label] = exp(xlabel) / sum - 1;

    for (int layer_index = grad.size() - 1; layer_index >= 0; layer_index--) { //process layer of weights in reverse order
        vector<vector<double> >& mat = grad[layer_index];
        const Layer& current_layer = layers[layer_index];
        assert(mat.size() == current_layer.weights.h && mat[0].size() == current_layer.weights.w + 1);
        assert(mat.size() == calced.size());
        int h = mat.size(), w = mat[0].size();
        vector<double> new_calced(w - 1, 0.0);

        //Worsed case try to use SIMD here ??? --- ooh yessir :)
        for (int i = 0; i < h; i++) {
            size_t alignedJ = (w - 1) - ((w - 1) % 4);
            const double sig_der_calced = sigmoid_derivative(calced[i]);
            __m256d sig_der_vec = _mm256_set_pd(sig_der_calced, sig_der_calced, sig_der_calced, sig_der_calced);
            for (int j = 0; j < alignedJ; j += 4) {
                __m256d neuronsvec = _mm256_loadu_pd(&neurons_activation[layer_index][j]);
                __m256d weightsvec = _mm256_loadu_pd(&current_layer.weights.elements[i][j]);

                vecadd_pd(&mat[i][j], _mm256_mul_pd(neuronsvec, sig_der_vec));
                vecadd_pd(&new_calced[j], _mm256_mul_pd(weightsvec, sig_der_vec));
            }
            for (int j = alignedJ; j < w - 1; j++) { //last one is the bias -- original loop
                mat[i][j] += neurons_activation[layer_index][j] * sig_der_calced; //Derivative with respect to weight
                new_calced[j] += current_layer.weights.elements[i][j] * sig_der_calced; //Derivative with respect to neuron value --- only used to propagate backwards
            }
            mat[i][w - 1] += sig_der_calced; //derivative with respect to bias
        }
        calced = new_calced;
    }
}

#endif