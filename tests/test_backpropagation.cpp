#include <iomanip>
#include "tests.hpp"


void test_bp_small_cases() {
	datalist data = {
		make_pair(Matrix<int>({10, 20}), 1),
		make_pair(Matrix<int>({30, 10}), 0),
		make_pair(Matrix<int>({20, 20}), 1),
		make_pair(Matrix<int>({10, 10}), 0),
		make_pair(Matrix<int>({0, 30}), 1)
	};
	cout << "Testing AI running on a small basic data set. Make sure the cost function is decreasing over time!!" << endl;
	DigitNetwork SmallAI({ 2, 4, 2 }, .01); // 2 - 4 - 2
	SmallAI.train(data, 300);

	auto neurons_activation = SmallAI.forward_prop(data[0].first);
	auto probs = SmallAI.analyze(data[0].first);
	cout << "Test AI trained, the output from the first data point: " << endl;
	cout << "\t neurons activation, layers are horizontal: " << endl;
	print_vecvec(neurons_activation);
	cout << "\t output pprobabilities: " << endl;
	print_vecvec({probs});
	cout << "the True value is" << data[0].second << endl;
}

/*Two simple cases tested manually with old back_prop. Everything appears correct!!! Could verify this using tensorflow.*/
void test_known_grad() {
	
	Matrix<double> input_neurons({ 1, 2 });
	Matrix<double> weights({ {2, 1}, {3, 7} });

	//Layer
	Layer free_layer(2, 2);
	int label = 0;
	free_layer.weights = weights;
	free_layer.biases = Matrix<double>({ 0,0 });
	vector<Layer> layers = { free_layer };

	vector<Matrix<double> > grad; //One layer
	grad.push_back(Matrix<double>(vector<vector<double> >(2, vector<double>(3, 0))));

	Matrix<double> output_neurons = free_layer.weights * input_neurons;
	for (int i = 0; i < output_neurons.h; i++) {
		output_neurons.elements[i][0] = sigmoid(output_neurons(i));
	}

	vector<Matrix<double> > neurons_activation = { input_neurons, output_neurons };
	auto probs = softmax(output_neurons.transpose().elements[0]);
	Matrix<double> matrix_probs(probs);
	datalist data = { make_pair(Matrix<int>({10, 20}), 1) };

	cout << "Layers' weights" << endl;
	for (auto lr : layers) {
		lr.weights.print(true);
	}
	cout << "Neurons activation" << endl;
	for (auto neurs : neurons_activation) {
		neurs.print(true);
	}
	cout << "Label: " << label << endl;

	back_prop_old_with_matrices(label, grad, neurons_activation, layers, matrix_probs);

	cout << "Gradient" << endl;
	for (auto gd : grad) {
		gd.print(true);
	}
}