#ifndef HEADER_UTILS
#define HEADER_UTILS

#include <cassert>
#include <vector>
#include <set>
#include <iostream>
#include <random>

using namespace std;
 
// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))
#define ROOT_FOLDER "/home/nils/Desktop/cpp_project"

#include "matrix.hpp"

typedef vector<pair<Matrix<int> , int> > datalist;
#include "network.hpp"


void tool();
datalist read_training_batch(int batch_size = 60000);
datalist sample_data(const datalist &data, int sample_size);
datalist read_test_data();

#endif