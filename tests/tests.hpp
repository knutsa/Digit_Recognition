#ifndef HEADER_TEST
#define HEADER_TEST
#include <chrono>
#include <immintrin.h>
#include "../src/utils.hpp"

//Experimental code

/*Add to_add to the doubles stored at p*/
inline void vecadd_pd(double* p, __m256d to_add) {
    __m256d loaded = _mm256_loadu_pd(p);
    __m256d updated = _mm256_add_pd(loaded, to_add);
    _mm256_storeu_pd(p, updated);
}
/*Attempt to make this run a little faster*/
inline void back_prop_no_matrices_simded(const int label, vector<vector<vector<double> > >& grad, const vector<vector<double> >& neurons_activation, const vector<Layer>& layers, const vector<double> output_probabilities) {
    /*Adds contribution of img to grad from img. Grad is a Matrix of the format [weight matrix | biases column], minimizing use of matrices to try and reduce computation time.*/
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

        //Worsed case try to use SIMD here ??? --- didn't work so good :(
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

//Actual test stuff
inline void print_vecvec(vector<vector<double> > m) {
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            cout << m[i][j] << ", ";
        }
        cout << endl;
    }
}

void test_matrix();
void test_bp_small_cases();
void test_known_grad();
void test_optimized_backprop();
#endif //header TEST