#include "tests.hpp"

using namespace std;


void test_matrix(){

    vector<vector<double> > data1 = {{1,2}, {3, 4}};
    vector<vector<double> > data2 = {{5, 6}, {7, 8}};
    vector<vector<double> > data3 = {{5, 6}, {7, 8}, {9, 10}};
    

    Matrix<double> A(data1);
    Matrix<double> B(data2);
    Matrix<double> C(data3);
    
    Matrix<double> S = A+B;
    for(int i = 0;i<2;i++){
        for(int j = 0;j<2;j++){
            assert(S.elements[i][j] == A.elements[i][j] + B.elements[i][j]);
        }
    }

    Matrix<double> P = C*A;
    assert(P.elements[0][0] == 23);
    assert(P.elements[2][0] == 39);
    
    cout << "Matrix test OK!" << endl;

    
}