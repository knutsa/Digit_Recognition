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

