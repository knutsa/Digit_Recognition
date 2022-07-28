#include <iostream>
#include <vector>
#include "../src/utils.hpp"

using namespace std;


template<typename dType>
void print_mat(Matrix<dType> mat){
    int h = mat.h, w = mat.w;
    for(int i = 0;i<h;i++){
        for(int j = 0;j<w;j++){
            std::cout << mat.elements[i][j]<< ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void main_test(){

    vector<vector<float> > data1 = {{1,2}, {3, 4}};
    vector<vector<float> > data2 = {{5, 6}, {7, 8}};
    vector<vector<float> > data3 = {{5, 6}, {7, 8}, {9, 10}};
    

    Matrix<float> A(data1);
    Matrix<float> B(data2);
    Matrix<float> C(data3);
    
    Matrix<float> S = A+B;
    for(int i = 0;i<2;i++){
        for(int j = 0;j<2;j++){
            assert(S.elements[i][j] == A.elements[i][j] + B.elements[i][j]);
        }
    }

    Matrix<float> P = C*A;
    assert(P.elements[0][0] == 23);
    assert(P.elements[2][0] == 39);
    
    cout << "Mat product" << endl;
    print_mat<float>(C*A);

    cout << "ALL TESTS OK" << endl;
    
}