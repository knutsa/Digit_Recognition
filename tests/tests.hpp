#ifndef HEADER_TEST
#define HEADER_TEST
#include <chrono>
#include <immintrin.h>
#include "../src/utils.hpp"

//Experimental code

//Old backprop, using matrices
/*Adds contribution of img to grad from img. Grad is a Matrix of the format [weight matrix | biases column]*/
inline void back_prop_old_with_matrices(const int label, vector<Matrix<double> >& grad, const vector<Matrix<double> >& neurons_activation, const vector<Layer>& layers, const Matrix<double> output_probabilities) {
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
        vector<double> new_calced(mat.w - 1, 0);

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

//Actual test stuff

void test_matrix();
void test_bp_small_cases();
void test_known_grad();
void test_optimized_backprop();
#endif //header TEST