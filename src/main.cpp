#include "utils.hpp"
//#include "../tests/test.hpp"

#include <iostream>

using namespace std;

int main(){

    datalist data = read_training_batch(1);
    tool();
    cout << "hallår" << endl;
    // cout << data[0].first.elements[10][11] << ' ' << data[0].second << endl;

    DigitNetwork AI(1.0);

    cout << "Training data " << endl;
    AI.train(data);

    cout << "AI trained" << endl;
    
    AI.analyze(data[0].first).print();

    // main_test();
}