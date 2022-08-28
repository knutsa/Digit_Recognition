#ifndef MATRIX_H
#define MATRIX_H

#include "utils.hpp"

template <class dType>
class Matrix{
    // 2D matrix with implementation for multiplication, addition, scalar multiplication, dType numeric, i.e long, int double et.c
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
    Matrix(vector<dType> arr, bool as_row = 0): h(arr.size()), w(1){
        vector<vector<dType> > inp(arr.size());
        if(as_row){
            inp = {arr};
        } else {
            for(int i = 0;i<arr.size();i++){
                inp[i] = {arr[i]};
            }
        }

        elements = inp;
    }
    Matrix(){}

    Matrix<dType> columnize(){ //Create Matrix which has the same data stored in column shape
        vector<vector<dType> > reshaped_data(h*w, vector<dType>(1));
        for(int i = 0;i<h;i++){
            for(int j = 0;j<w;j++){
                reshaped_data[i*w + j][0] = elements[i][j];
            }
        }

        return Matrix<dType>(reshaped_data);
        
    }

    void print(bool print_all = false){
        int upper_limit = 5;
        if(print_all)
            upper_limit = max(h, w);
        cout << "Matrix " << endl;
        for(int i = 0;i<min(h, upper_limit);i++){
            for(int j = 0;j<min(w, upper_limit);j++){
                cout << elements[i][j] << ' ';
            }
            cout << endl;
        }
        cout << endl;
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
    dType operator[](int index){
        assert(w == 1);
        return elements[index][0];
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

    Matrix<dType> operator-(const Matrix<dType> &other){
        assert(other.h == h);
        assert(other.w == w);
        vector<vector<dType> > res(h, vector<dType>(w));
        for(int i = 0;i<h;i++){
            for(int j = 0;j<w;j++){
                res[i][j] = elements[i][j] - other.elements[i][j];
            }
        }

        return Matrix<dType>(res);
    }

    dType operator[](pair<int, int> indeces){
        int i = indeces.first, j = indeces.second;
        return elements[i][j];
    }
};


#endif