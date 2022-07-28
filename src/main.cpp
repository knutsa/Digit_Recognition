#include "utils.hpp"
#include "../tests/test.hpp"

#include <iostream>

using namespace std;

int main(){

    vector<pair<Matrix<int>, int> > data = read_training_batch();
    tool();

    cout << data[0].first.elements[10][11] << ' ' << data[0].second << endl;

    // main_test();
}