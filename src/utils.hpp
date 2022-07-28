#ifndef HEADER_UTILS
#define HEADER_UTILS

#include <cassert>
#include <vector>
 
// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))
#define ROOT_FOLDER "/home/nils/Desktop/cpp_project"

using namespace std;

#include "matrix.hpp"

void tool();
vector<pair<Matrix<int>, int > > read_training_batch();
vector<pair<Matrix<int>, int > > read_test_data();


#endif