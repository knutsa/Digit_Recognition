#include "utils.hpp"

using namespace std;

int main(){

    cout << "Reading data" << endl;
    datalist full_data_set = read_training_batch();
    cout << "Done reading " << full_data_set.size() << " images" << endl;

    //full_data_set = preprocess(full_data_set);
    DigitNetwork AI({784, 60, 60, 10}, 2.0);
    cout << "Neural Network initialized with random weights. Network size is 784 x 50 x 50 x 10" << endl;
    cout << "Initial cost: ";
    auto res = AI.cost_function(full_data_set);
    double initial_cost = res.first, initial_accuracy = res.second;
    cout << fixed << setprecision(10) << initial_cost << " initial accuracy: " << initial_accuracy << "%" << endl;
  
    //Training
    AI.train(full_data_set, 15, 100);
    AI.scale_learning(.75);
    AI.train(full_data_set, 15, 100);
    AI.scale_learning(0.75);
    AI.train(full_data_set, 20, 100);

    cout << "AI is trained." << endl;

    cout << string(40, '=') << endl;
    cout << "Performance summary: " << endl;

    cout << "Reminding, initial cost was: " << initial_cost << " and initial accuracy was " << initial_accuracy << " %" << endl;
    auto final_res = AI.cost_function(full_data_set);
    double final_cost = final_res.first, final_accuracy = final_res.second;
    cout << "Final Cost is: "  << final_cost << " and the network has a training accuracy of: " << final_accuracy << " %" << endl;
    
    datalist test_data = read_test_data();
    auto test_res = AI.cost_function(test_data);
    cout << "Test cost is: " << test_res.first << " and test accuracy is: " << test_res.second << "%" << endl;
}