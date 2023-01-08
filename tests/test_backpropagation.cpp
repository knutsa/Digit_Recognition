#include <iomanip>
#include "tests.hpp"


void test_bp_small_cases(){
	/*The value of the cost function is not reducing over time!!!?! */
	datalist data = {
		make_pair(Matrix<int>({10, 20}), 1),
		make_pair(Matrix<int>({30, 10}), 0),
		make_pair(Matrix<int>({20, 20}), 1),
		make_pair(Matrix<int>({10, 10}), 0),
		make_pair(Matrix<int>({0, 30}), 1) 
	};//, make_pair(Matrix<int>({ 20, 30 }), 5)};
	DigitNetwork SmallAI({ 2, 4, 2 }, .1); // 2- 2
	SmallAI.train(data, 30);

	auto probs = SmallAI.analyze(data[0].first);
	probs.print(true);
	cout << "True value" << data[0].second << endl;
}

void test_known_grad() {
	/*Two simple cases tested. Everything appears correct!?!!! Could verify this using tensorflow.*/
	
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
		output_neurons.elements[i][0] = sigmoid(output_neurons[i]);
	}

	vector<Matrix<double> > neurons_activation = { input_neurons, output_neurons };
	auto probs = softmax(output_neurons);
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

	back_prop(label, grad, neurons_activation, layers, probs);

	cout << "Gradient" << endl;
	for (auto gd : grad) {
		gd.print(true);
	}
}