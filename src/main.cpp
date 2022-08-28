#include "utils.hpp"
//#include "../tests/test.hpp"

#include <iostream>

using namespace std;

int main(){

    datalist full_data_set = read_training_batch(60000);
    tool();

    DigitNetwork AI(10);
    cout << "Initial cost " << AI.cost_function(full_data_set) << endl;

    cout << "Training data " << endl;
    
    //Training
    for(int i = 0;i<100;i++){
        cout << "Iteration " << i << endl;
        datalist data_batch = sample_data(full_data_set, 300);
        AI.train(data_batch);
    }

    cout << "AI trained" << endl;

    for(int i = 0;i<5;i++){
        cout << "Predictions for image nr " << i << endl;
        auto res = AI.analyze(full_data_set[0].first);
        res[res.size()-1].print(true);
    }    

    cout << "Cost "  << AI.cost_function(full_data_set) << endl;
    // main_test();
}