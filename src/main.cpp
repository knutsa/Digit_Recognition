#include "utils.hpp"

using namespace std;

int main(){

    cout << "Reading data" << endl;
    datalist full_data_set = read_training_batch(60000);
    cout << "Done reading " << full_data_set.size() << " images" << endl;

    full_data_set = preprocess(full_data_set);
    DigitNetwork AI({196, 50, 50, 10}, .01);
    cout << "Neural Network innitialized with random weights." << endl;
    cout << "Initial cost:" << endl;
    double initial_cost = AI.cost_function(full_data_set);
    cout << fixed << setprecision(5) << initial_cost << endl;
    
    //Training
    AI.train(full_data_set, 30, 200);

    cout << "AI trained." << endl;

    for(int i = 0;i<5;i++){
        auto img = full_data_set[i].first;
        auto label = full_data_set[i].second;
        cout << "Predictions for image nr " << i << " (true value is " << label << ")" << endl;
        auto res = AI.analyze(img);
        const Matrix<double> output_neurons = res[res.size() - 1];
        for (int k = 0; k < 10; k++) {
            cout << k << ": " << output_neurons.elements[k][0] << endl;
        }
    }    

    cout << "Reminding, initial cost was: " << initial_cost << endl;
    cout << "Final Cost is: "  << AI.cost_function(full_data_set) << endl;
    // main_test();
}