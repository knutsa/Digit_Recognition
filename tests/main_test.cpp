#include "tests.hpp"

void meassure_times() {
	DigitNetwork AI({ 100,100,10 }, .01);
	vector<Matrix<double> > neurons_activation = { zero_matrix<double>(100,1), zero_matrix<double>(100, 1), zero_matrix<double>(10,1) };
	vector<Matrix<double> > grad = { zero_matrix<double>(100,101), zero_matrix<double>(10, 101) };
	auto probs = zero_matrix<double>(10, 1);
	probs.elements[0][0] = 1;

	auto start = chrono::high_resolution_clock::now();
	datalist data = read_training_batch();
	auto stop1 = chrono::high_resolution_clock::now();

	for (auto p : data) {
		auto img = p.first;
		int label = p.second;
		auto res = AI.forward_prop(img);
	}
	auto stop2 = chrono::high_resolution_clock::now();
	for (auto p : data) {
		auto img = p.first;
		int label = p.second;
		auto res = back_prop(label, grad, neurons_activation, probs);
	}
	
}


int main() {
	//test_matrix();


	test_bp_small_cases();

	//test_known_grad();
}