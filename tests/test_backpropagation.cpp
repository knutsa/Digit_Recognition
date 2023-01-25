#include <iomanip>
#include "tests.hpp"


void test_bp_small_cases() {
	datalist data = { //big x0 -> label = 0, big x1 -> label = 1
		make_pair(Matrix<int>({200, 50}), 0),
		make_pair(Matrix<int>({20, 150}), 0),
		make_pair(Matrix<int>({250, 60}), 0),
		make_pair(Matrix<int>({80, 230}), 0),
		make_pair(Matrix<int>({220, 30}), 0)
	};
	DigitNetwork SmallAI({ 2, 4, 2 }, .01); // 2 - 4 - 2
	cout << "Testing AI running on a small basic data set. Make sure the cost function is decreasing over time! Hope it works now!" << endl;
	cout << "initial weights" << endl;
	SmallAI.layers[0].weights.print(true);

	SmallAI.train(data, 100);

	auto neurons_activation = SmallAI.forward_prop(data[0].first);
	auto probs = SmallAI.analyze(data[0].first);
	cout << "Test AI trained, the output from the first data point: " << endl;
	cout << "\t neurons activation, layers are horizontal: " << endl;
	print_vecvec(neurons_activation);
	cout << "\t output pprobabilities: " << endl;
	print_vecvec({probs});
	cout << "the True value is" << data[0].second << endl;

	cout << "AI weights " << endl;
	SmallAI.layers[0].weights.print(true);
	SmallAI.layers[0].biases.print(true);
}

/*Two simple cases tested manually with old back_prop. Everything appears correct!!! Could verify this using tensorflow.*/
void test_known_grad() {
}