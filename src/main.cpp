#include "utils.hpp"

using namespace std;

int main(){

    cout << "Reading data" << endl;
    datalist full_data_set = read_training_batch();
    cout << "Done reading " << full_data_set.size() << " images" << endl;

    //full_data_set = preprocess(full_data_set);
    DigitNetwork AI({784, 50, 50, 10}, .1);
    cout << "Neural Network initialized with random weights. Network size is 784 x 50 x 50 x 10" << endl;
    cout << "Initial cost:" << endl;
    auto res = AI.cost_function(full_data_set);
    double initial_cost = res.first, initial_accuracy = res.second;
    cout << fixed << setprecision(10) << initial_cost << endl;
    
    //Training
    AI.train(full_data_set,100, 300);

    cout << "AI trained." << endl;

    for(int i = 0;i<5;i++){
        auto img = full_data_set[i].first;
        auto label = full_data_set[i].second;
        cout << "Predictions for image nr " << i << " (true value is " << label << ")" << endl;
        auto res = AI.analyze(img);
        for (int k = 0; k < 10; k++) {
            cout << k << ": " << res(k) << endl;
        }
    }

    cout << "Reminding, initial cost was: " << initial_cost << " and initial accuracy was " << initial_accuracy << " %" << endl;
    auto final_res = AI.cost_function(full_data_set);
    double final_cost = final_res.first, final_accuracy = final_res.second;
    cout << "Final Cost is: "  << final_cost << " and the network has a training accuracy of: " << final_accuracy << " %" << endl;
    // main_test();
}