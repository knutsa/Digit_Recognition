#ifndef NETWORK_H
#define NETWORK_H

#include "utils.hpp"

#define sigmoid(x) 1 / (1 + exp(-x))
#define sigmoid_derivative(x) 1 / ( (exp(x/2) + exp(-x/2)) *(exp(x/2)+exp(-x/2)) )
#define sigmoid_der_fromsqueezed(y) y*(1-y) //y' = y*(1-y)

#define L2 0
#define CROSS_CATEGORICAL_ENTROPY 1

class Layer {
public:
    Matrix<double> biases; //biases used to transform to itself
    Matrix<double> weights; //weights used to transform from previous layer to itself

    Layer(int prev_sz, int sz) :
        biases(Matrix<double>(vector<double>(sz, .0)))
    {
        weights = Matrix<double>(vector<vector<double> >(sz, vector<double>(prev_sz)));

        default_random_engine generator(time(0));
        normal_distribution<double> distribution(0, 1); //This part is probably quite important. Initialize with the right weights and biases!! maybe read up a bit on this
        for(int i = 0;i<weights.h;i++){
            for(int j = 0;j<weights.w;j++){
                weights.elements[i][j] =  distribution(generator);
            }
        }
    }
    /*Initialize layer from matrix in format [weights | biases]*/
    Layer(const vector<vector<double> >& params) {
        size_t h = params.size(), w = params[0].size() - 1;
        weights = zero_matrix<double>(h, w);
        biases = zero_matrix<double>(h, 1);

        for (int i = 0; i < h; i++) {
            assert(params[i].size() == w+1);
            for(int j = 0; j < w;j++) {
                weights.elements[i][j] = params[i][j];
            }
            biases.elements[i][0] = params[i][w];
        }
    }
};


class DigitNetwork{
private:
    double learning_rate;
    char loss;
    /*Perform one epoch*/
    void epoch(const datalist &data, int batch_size = 100);
public:
    /*
        Neural Network class to recognize digit images of 28 x 28 pixels.
        By default Three dense layers 784 - 50 - 50 - 10
    */

    vector<Layer> layers;

    /*
        :param neuron_sizes -- network dimensions
        :param learning_r -- learning_rate, constant for all weights and epochs
        :param loss -- either of the defined macros 'L2' for mean square or 'CROSS_CATEGORICAL_ENTROPY'
    */
   DigitNetwork(vector<int> neuron_sizes = {784, 50, 50, 10}, double learning_r = .01, char loss = L2) {
       assertm(loss == L2 || loss == CROSS_CATEGORICAL_ENTROPY, "Loss must be any of 'L2' or 'CROSS_CATEGORICAL_ENTROPY' ");
     this->loss = loss;
     layers = {};
     for (int i = 0; i < neuron_sizes.size() - 1; i++) {
         int from = neuron_sizes[i], to = neuron_sizes[i + 1];
         layers.push_back(Layer(from, to));
     }
     learning_rate = learning_r;
   }
   /*Initilize Network with a chosen (supposedly read) set of layers*/
   DigitNetwork(vector<Layer> chosen_layers, char loss = L2) : layers(chosen_layers), loss(loss) {}

   /*Calculate neuron activations -- i.e forward propagation, floating point number between -0.5 to 0.5 for each neuron */
   vector<vector<double> > inline forward_prop(const Matrix<int> &img);

   vector<double> inline analyze(const Matrix<int>& img);

   void train(datalist data, int epochs = 30, int batch_size = 100);
   /*Fit network to given data*/

   /*
    Evaluatee cost function and accuracy for the given data. The loss function used is the CrossCategoricalEntropy

    :return cost, accuracy (%)
    */
   pair<double, double> cost_function(const datalist &data);

   string loss_rep() const {
       if (this->loss == L2)
           return "mean square";
       if (this->loss == CROSS_CATEGORICAL_ENTROPY)
           return "CROSS_CATEGORICAL_ENTROPY";
       return "!!invalid loss!!!";
   }
   char get_loss() const {
       return this->loss;
   }
   /*Multiply learning rate by scale*/
   void scale_learning(double scale = .5) {
       this->learning_rate = this->learning_rate * scale;
   }

};

//inline methods
vector<vector<double> > inline DigitNetwork::forward_prop(const Matrix<int> &img){
    
    vector<double> input_neurons; //convert img to neurons, could include rescaling here!! :)
    for (int i = 0; i < img.h; i++) {
        for (int j = 0; j < img.w; j++)
            input_neurons.push_back((double) img.elements[i][j]);
    }
    Matrix<double> input_mat(input_neurons);
    input_mat = input_mat * (1.0 / 256.0);
    vector<vector<double> > neuron_activations;
    assertm(input_mat.h == this->layers[0].weights.w, "Network Neurons sizes are mismatching the input size!!!");

    neuron_activations.push_back(input_mat.transpose().elements[0]); //copy over
    for(auto lr : this->layers){
        input_mat = lr.weights * input_mat + lr.biases;
        vector<double> neuron_activation(input_mat.h);
        for(int i = 0;i<input_mat.h;i++){
            double squeezed = sigmoid(input_mat(i));
            neuron_activation[i] = input_mat.elements[i][0] = squeezed;
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
 /*Derivative of Cross categorical entropy cost function with respect to output neurons (x), cost = -log(y[label]) */
vector<double> inline softmax_cost_derivative(const vector<double>& x, const vector<double>& y, int label) {
    double sum = 0.0;
    assert(x.size() == y.size());
    assert(label >= 0 && label < x.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++)
        sum += exp(x[i]);
    for (int i = 0; i < x.size(); i++)
        res[i] = exp(x[i]) / sum;
    res[label] = exp(x[label]) / sum - 1;

    return res;
}
/*Calculate derivative of the L2 cost fuunction with respect to output_neurons*/
vector<double> inline l2_cost_derivative(const vector<double>& x, int label) {
    assert(label >= 0 && label < x.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++)
        res[i] = 2 * x[i];
    res[label] = 2 * (x[label] - 1);

    return res;
}

vector<double> DigitNetwork::analyze(const Matrix<int>& img) {
    auto neurons_activation = this->forward_prop(img);
    auto &output_neurons = *(neurons_activation.end() - 1);
    //Run logits through softmax
    switch (this->loss) {
    case L2:
        return output_neurons;
    case CROSS_CATEGORICAL_ENTROPY:
        return softmax(output_neurons);
    default:
        assertm(1 == 0, "invalid loss used");
    }
    return {0};
}

/*Add to_add to the doubles stored at p (simd version of +=)*/
inline void vecadd_pd(double* p, __m256d to_add) {
    __m256d loaded = _mm256_loadu_pd(p);
    __m256d updated = _mm256_add_pd(loaded, to_add);
    _mm256_storeu_pd(p, updated);
}

//This function is the bottleneck of the algorithm, it has been modified to run a little faster
/*Adds contribution of img to grad from img. Grad is a Matrix of the format [weight matrix | biases column], minimizing use of matrices to try and reduce computation time.*/
inline void back_prop(const int label, vector<vector<vector<double> > >& grad, const vector<vector<double> >& neurons_activations, const vector<Layer>& layers, const vector<double>& output_neuron_derivatives) {
    assert(neurons_activations.size() == grad.size() + 1); //one layer between each group of neurons
    const auto& output_neurons = *(neurons_activations.end() - 1);
    assert(output_neuron_derivatives.size() == output_neurons.size());
    vector<double> calced = output_neuron_derivatives;

    for (int layer_index = grad.size() - 1; layer_index >= 0; layer_index--) {
        //LOOP INVARIANT: calced is vector of derivatives of the cost function with respect to the neurons coming out of the layer being processed, i.e neurons[layer_index+1]
        vector<vector<double> >& mat = grad[layer_index];
        const Layer& current_layer = layers[layer_index];
        size_t h = mat.size(), w = mat[0].size();
        assert(h == current_layer.weights.h && w == current_layer.weights.w + 1);
        assert(h == calced.size());
        vector<double> new_calced(w - 1, 0.0);

        for (int i = 0; i < h; i++) { //Optimized with SIMD to decrease time -- This is the bottleneck
            size_t alignedJ = (w - 1) - ((w - 1) % 4);
            const double sig_der_calced = sigmoid_der_fromsqueezed(neurons_activations[layer_index+1][i]) * calced[i];
            __m256d sig_der_vec = _mm256_setr_pd(sig_der_calced, sig_der_calced, sig_der_calced, sig_der_calced);
            for (int j = 0; j < alignedJ; j += 4) {
                __m256d neuronsvec = _mm256_loadu_pd(&neurons_activations[layer_index][j]);
                __m256d weightsvec = _mm256_loadu_pd(&current_layer.weights.elements[i][j]);

                vecadd_pd(&mat[i][j], _mm256_mul_pd(neuronsvec, sig_der_vec));
                vecadd_pd(&new_calced[j], _mm256_mul_pd(weightsvec, sig_der_vec));
            }
            for (int j = alignedJ; j < w - 1; j++) { //last one is the bias -- original loop
                mat[i][j] += neurons_activations[layer_index][j] * sig_der_calced; //Derivative with respect to weight
                new_calced[j] += current_layer.weights.elements[i][j] * sig_der_calced; //Derivative with respect to neuron value --- only used to propagate backwards
            }
            mat[i][w - 1] += sig_der_calced; //derivative with respect to bias
        }
        calced = new_calced;
    }
}

//Saving the model

void store_model(const DigitNetwork& AI, double loss, double accuracy, string filename="saved_model");
DigitNetwork load_model(string filename="saved_model");

#endif