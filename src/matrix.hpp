#ifndef MATRIX_H
#define MATRIX_H

#include "utils.hpp"

template <class dType>
class Matrix{
    // 2D matrix with implementation for multiplication, addition, scalar multiplication, dType numeric, i.e long, int float et.c
public:
    vector<vector<dType> > elements;
    int h;
    int w;
    //It seems as template code must be inline - since it is included at multiple spots

    Matrix(vector<vector<dType> > inp): h(inp.size()), w(inp[0].size()) {
        elements = inp;
        for(int i = 0;i<h;i++){
            assert(elements[i].size() == w);
        }
    }

    Matrix<dType> operator+(const Matrix<dType> &other){
        assert(other.h == h);
        assert(other.w == w);
        vector<vector<dType> > res(h, vector<dType>(w));
        for(int i = 0;i<h;i++){
            for(int j = 0;j<w;j++){
                res[i][j] = elements[i][j] + other.elements[i][j];
            }
        }

        return Matrix<dType>(res);
    }

    Matrix<dType> operator*(const Matrix<dType> &other){
        assert(w == other.h);
        vector<vector<dType> > res(h, vector<dType>(other.w));

        for(int i = 0;i<h;i++){
            for(int j = 0;j<other.w;j++){
                dType sum = 0;
                for(int k = 0;k<w;k++){
                    sum += elements[i][k]*other.elements[k][j];
                }
                res[i][j] = sum;
            }
        }
        
        return Matrix<dType>(res);
    }

    Matrix<dType> operator*(dType scalar){
        vector<vector<dType> > res(h, vector<dType>(w));

        for(int i = 0;i<h;i++){
            for(int j = 0;j<w;j++){
                res[i][j] = scalar * elements[i][j];
            }
        }
        
        return Matrix<dType>(res);
    }
};

#endif