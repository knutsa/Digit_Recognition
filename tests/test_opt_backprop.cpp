#include "tests.hpp"


void test_optimized_backprop() {
	DigitNetwork AI({ 10,10 }, .01);
	vector<Matrix<double> > neurons_activation = { random_matrix<double>(10,1), random_matrix<double>(10, 1)};
	vector<Matrix<double> > grad = {zero_matrix<double>(10, 11) };
	auto probs = zero_matrix<double>(10, 1);
	probs.elements[0][0] = 1;

	vector<vector<double> > no_matrix_neurons_activation;
	for (auto m : neurons_activation) { no_matrix_neurons_activation.push_back(m.transpose().elements[0]); }
	vector<vector<vector<double> > > no_matrix_grad;
	for (auto m : grad) { no_matrix_grad.push_back(zero_matrix<double>(m.h, m.w).elements); }
	vector<double> no_matrix_prob = probs.transpose().elements[0];
	int label = 1;

	back_prop(label, grad, neurons_activation, AI.layers, probs);
	grad[0].print(true);
	for (int l_ind = 0; l_ind < grad.size(); l_ind++) {
		size_t h = grad[l_ind].h, w = grad[l_ind].w;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				assert(grad[l_ind](i, j) != no_matrix_grad[l_ind][i][j]);
			}
		}
	}
	back_prop_no_matrices_simded(label, no_matrix_grad, no_matrix_neurons_activation, AI.layers, no_matrix_prob);
	cout << "No matrix grad" << endl;
	print_vecvec(no_matrix_grad[0]);
	double epsilon = 1e-7;
	for (int l_ind = 0; l_ind < grad.size(); l_ind++) {
		size_t h = grad[l_ind].h, w = grad[l_ind].w;
		int counter = 0;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (abs(grad[l_ind](i, j) - no_matrix_grad[l_ind][i][j]) > epsilon) {
					cout << "Values different at " << i << ", " << j << endl;
					cout << "Values are: " << grad[l_ind](i, j) << " and " << no_matrix_grad[l_ind][i][j] << endl;
					counter++;
					if (counter == 10)
						return;
				}
			}
		}
	}
	cout << "All values equal to margin " << epsilon << endl;
	cout << "Enf of test optimizec backprop" << endl;
}