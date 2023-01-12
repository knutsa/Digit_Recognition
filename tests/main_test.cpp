#include "tests.hpp"
#include <string>

void meassure_times() {
	const int num_imgs = 1000;
	cout << "Measuring time, using " << num_imgs << " images" << endl;
	DigitNetwork AI({ 784, 50, 50, 10 }, .01);
	vector<Matrix<double> > neurons_activation = { random_matrix<double>(784,1), random_matrix<double>(50, 1), random_matrix<double>(50, 1), random_matrix<double>(10,1) };
	vector<Matrix<double> > grad = { random_matrix<double>(50,785), random_matrix<double>(50, 51), random_matrix<double>(10, 51)};
	auto probs = zero_matrix<double>(10, 1);
	probs.elements[0][0] = 1;

	vector<vector<double> > no_matrix_neurons_activation;
	for (auto m : neurons_activation) { no_matrix_neurons_activation.push_back(m.transpose().elements[0]); }
	vector<vector<vector<double> > > no_matrix_grad;
	for (auto m : grad) {
		vector<vector<double> > els(m.h, vector<double>(m.w));
		for (int i = 0; i < m.h; i++) {
			for (int j = 0; j < m.w; j++) {
				els[i][j] = m(i, j);
			}
		}
		no_matrix_grad.push_back(els);
	}
	vector<double> no_matrix_prob = probs.transpose().elements[0];
	/*datalist artificial_data;
	for (int i = 0; i < 1000; i++) {
		artificial_data.push_back(make_pair(zero_matrix<int>(100, 1), 1));
	}*/

	auto reading0 = chrono::high_resolution_clock::now();
	datalist data = read_training_batch(num_imgs);
	auto reading1 = chrono::high_resolution_clock::now();
	auto reading_data_time = chrono::duration_cast<chrono::nanoseconds>(reading1 - reading0).count();
	cout << "Done reading, " << reading_data_time *1e-9 << "s" << endl;

	auto forward0 = chrono::high_resolution_clock::now();
	for (auto p : data) {
		auto img = p.first;
		int label = p.second;
		auto res = AI.forward_prop(img);
	}
	auto forward1 = chrono::high_resolution_clock::now();
	auto forward_time = chrono::duration_cast<chrono::nanoseconds>(forward1 - forward0).count();
	cout << "Done forward, " << forward_time * 1e-9 <<  "s" << endl;

	auto back0 = chrono::high_resolution_clock::now();
	{
		size_t processed = 0;
		for (unsigned int data_index = 0;data_index<data.size();data_index++) {
			auto img = data[data_index].first;
			int label = data[data_index].second;
			back_prop_old_with_matrices(label, grad, neurons_activation, AI.layers, probs);
			//back_prop_no_matrices(label, no_matrix_grad, no_matrix_neurons_activation, AI.layers, no_matrix_prob);
		}
	}
	auto back1 = chrono::high_resolution_clock::now();
	auto back_time = chrono::duration_cast<chrono::nanoseconds>(back1 - back0).count();
	cout << "Done backward old, " << back_time * 1e-9 << " s" << endl;
	
	auto no_matrix_back0 = chrono::high_resolution_clock::now();
	{
		size_t processed = 0;
		for (unsigned int data_index = 0;data_index<data.size();data_index++) {
			auto img = data[data_index].first;
			int label = data[data_index].second;
			back_prop(label, no_matrix_grad, no_matrix_neurons_activation, AI.layers, no_matrix_prob);
		}
	}
	auto no_matrix_back1 = chrono::high_resolution_clock::now();
	auto no_matrix_back_time = chrono::duration_cast<chrono::nanoseconds>(no_matrix_back1 - no_matrix_back0).count();
	cout << "Attempted Optimized backward - using simd and less matrix classes, " << no_matrix_back_time * 1e-9 << " s" << endl;

	cout << string(40, '=') << endl;
	cout << "Results:" << endl;
	cout << "reading time " << reading_data_time * 1e-9 << " s" << endl;
	cout << "forward time " << forward_time * 1e-9 << " s" << endl;
	cout << "backward time " << back_time * 1e-9 << " s" << endl;
	cout << "opt  backward " << no_matrix_back_time * 1e-9 << " s" << endl;	

	//Assure results are the same
	double epsilon = 1e-12;
	assert(grad.size() == no_matrix_grad.size());
	for (int grad_index = 0; grad_index < grad.size(); grad_index++) {
		size_t h = grad[grad_index].h, w = grad[grad_index].w;
		int counter = 0;
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; j++) {
				if (abs(grad[grad_index](i, j) - no_matrix_grad[grad_index][i][j]) > epsilon) {
					cout << "Values different at " << i << ", " << j << endl;
					cout << "Values are: " << grad[grad_index](i, j) << " and " << no_matrix_grad[grad_index][i][j] << endl;
					counter++;
					if (counter == 10)
						return;
 				}
			}
		}
	}
	cout << "All values equal within " << epsilon << endl;
}


int main() {
	//test_matrix();


	//test_bp_small_cases();

	//test_known_grad();
	//test_optimized_backprop();
	meassure_times();

	return 0;
}