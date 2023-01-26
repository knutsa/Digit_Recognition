#include "tests.hpp"

void benchmaks() {
	DigitNetwork AI({ 784, 60, 60, 10 }, .01);
	vector<vector<double> > neurons_activation = { random_matrix<double>(1,784).elements[0], random_matrix<double>(1,60).elements[0],random_matrix<double>(1, 60).elements[0], random_matrix<double>(1, 10).elements[0] };
	vector<vector<vector<double> > > grad = { random_matrix<double>(60,785).elements, random_matrix<double>(60, 61).elements, random_matrix<double>(10, 61).elements };
	auto output_ders = random_matrix<double>(1, 10).elements[0];
	datalist data;

	auto reading0 = chrono::high_resolution_clock::now();
	data = read_training_batch();
	auto reading1 = chrono::high_resolution_clock::now();
	double reading_time = chrono::duration_cast<chrono::nanoseconds>(reading1 - reading0).count() * 1e-9;
	cout << "Done reading (" << reading_time << ")s" << endl;

	auto backward0 = chrono::high_resolution_clock::now();
	for (auto dp : data) {
		back_prop(3, grad, neurons_activation, AI.layers, output_ders);
	}
	auto backward1 = chrono::high_resolution_clock::now();
	double backward_time = chrono::duration_cast<chrono::nanoseconds>(backward1 - backward0).count() * 1e-9;
	cout << "Done backward (" << backward_time << ")s" << endl;

	auto forward0 = chrono::high_resolution_clock::now();
	for (auto dp : data) {
		const auto& img = dp.first;
		int label = dp.second;

		auto neurons = AI.forward_prop(img);
	}
	auto forward1 = chrono::high_resolution_clock::now();
	double forward_time = chrono::duration_cast<chrono::nanoseconds>(forward1 - forward0).count() * 1e-9;
	cout << "Done forward (" << forward_time << ")s" << endl;
}

//experimental -- occasionally crashes
/*Adds contribution of img to grad from img. Grad is a Matrix of the format [weight matrix | biases column], minimizing use of matrices to try and reduce computation time.*/
inline void back_prop_faster(const int label, vector<vector<vector<double> > >& grad, const vector<vector<double> >& neurons_activations, const vector<Layer>& layers, const vector<double>& output_neuron_derivatives) {
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
            const double sig_der_calced = sigmoid_der_fromsqueezed(neurons_activations[layer_index + 1][i]) * calced[i];
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