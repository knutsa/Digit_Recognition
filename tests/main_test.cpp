#include "tests.hpp"

void meassure_times() {
	DigitNetwork AI({ 100,100,10 }, .01);
	vector<Matrix<double> > neurons_activation = { zero_matrix<double>(100,1), zero_matrix<double>(100, 1), zero_matrix<double>(10,1) };
	vector<Matrix<double> > grad = { zero_matrix<double>(100,101), zero_matrix<double>(10, 101) };
	auto probs = zero_matrix<double>(10, 1);
	probs.elements[0][0] = 1;
	datalist data;
	for (int i = 0; i < 1000; i++) {
		data.push_back(make_pair(zero_matrix<int>(100, 1), 1));
	}


	auto start = chrono::high_resolution_clock::now();
	datalist useless = read_training_batch();
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
		back_prop(label, grad, neurons_activation, AI.layers, probs);
	}
	auto stop3 = chrono::high_resolution_clock::now();

	auto reading_data_time = chrono::duration_cast<chrono::nanoseconds>(stop1 - start);
	auto forward_time = chrono::duration_cast<chrono::nanoseconds>(stop2 - stop1);
	auto back_time = chrono::duration_cast<chrono::nanoseconds>(stop3 - stop2);

	cout << "reading time " << reading_data_time.count() * 1e-9 << " s" << endl;
	cout << "forward time " << forward_time.count() * 1e-9 << " s" << endl;
	cout << "backward time " << back_time.count() * 1e-9 << " s" << endl;
	
}


int main() {
	//test_matrix();


	//test_bp_small_cases();

	//test_known_grad();'
	meassure_times();

	return 0;
}