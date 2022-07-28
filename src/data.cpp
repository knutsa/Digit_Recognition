#include <iostream>
#include "utils.hpp"


using namespace std;


void tool(){
    cout << "This is the only non-inline util implemented yet. It is useless!!!" << endl;
}

int reverseInt(unsigned char *a)
{
	return ((((a[0] * 256) + a[1]) * 256) + a[2]) * 256 + a[3];
}

//Return vector of images with corresponding labels, digits 0-9
vector<pair<Matrix<int>, int > > read_training_batch(){
    vector<pair<Matrix<int>, int> > res;
    vector<Matrix<int>> imgs;
    /*
        Might add functionality to randomly choose subgroup of all 60000
    */

    {
        //Read images
        FILE * pFile = fopen(
            ROOT_FOLDER"/data/train-images-idx3-ubyte",
            "rb"
        );
        
        unsigned char inp[4];
        int num_read;
        num_read = fread(inp, 1, 4, pFile);
        int magic = reverseInt(inp);
        num_read = fread(inp, 1, 4, pFile);
        int num_imgs = reverseInt(inp);
        num_read = fread(inp, 1, 4, pFile);
        int h = reverseInt(inp);
        num_read = fread(inp, 1, 4, pFile);
        int w = reverseInt(inp);

        assert(magic == 2051 && h == 28 && w == 28);

        for(int n = 0;n<num_imgs;n++){
            unsigned int img[28][28];
            fread(img, 28, 28, pFile);
            vector<vector<int> > data(28, vector<int>(28));
            for(int i = 0;i<28;i++){
                for(int j = 0;j<28;j++){
                    data[i][j] = (int) img[i][j];
                }
            }
            imgs.push_back(Matrix<int>(data));
        }
        fclose(pFile);
    }
    {
        //Label images
        FILE * pFile = fopen(
            ROOT_FOLDER"/data/train-labels-idx1-ubyte",
            "rb"
        );

        unsigned char inp[4];
        int num_read;
        num_read = fread(inp, 1, 4, pFile);
        int magic = reverseInt(inp);
        num_read = fread(inp, 1, 4, pFile);
        int num_labels = reverseInt(inp);
        assert(num_labels == imgs.size() && magic == 2049);
        unsigned char labels[num_labels];
        fread(labels, 1, num_labels, pFile);

        for(int n = 0;n<num_labels;n++){
            res.push_back(pair<Matrix<int>, int>(imgs[n], (int) labels[n]));
        }
    }

    return res;
}

//Return vector of images with corresponding labels, digits 0-9
vector<pair<Matrix<int>, int > > read_test_data(){
    vector<pair<Matrix<int>, int> > res;
    vector<Matrix<int>> imgs;

    {
        //Read images
        FILE * pFile = fopen(
            ROOT_FOLDER"/data/t10k-images-idx3-ubyte",
            "rb"
        );
        
        unsigned char inp[4];
        int num_read;
        num_read = fread(inp, 1, 4, pFile);
        int magic = reverseInt(inp);
        num_read = fread(inp, 1, 4, pFile);
        int num_imgs = reverseInt(inp);
        num_read = fread(inp, 1, 4, pFile);
        int h = reverseInt(inp);
        num_read = fread(inp, 1, 4, pFile);
        int w = reverseInt(inp);

        assert(magic == 2051 && h == 28 && w == 28);

        for(int n = 0;n<num_imgs;n++){
            unsigned int img[28][28];
            fread(img, 28, 28, pFile);
            vector<vector<int> > data(28, vector<int>(28));
            for(int i = 0;i<28;i++){
                for(int j = 0;j<28;j++){
                    data[i][j] = (int) img[i][j];
                }
            }
            imgs.push_back(Matrix<int>(data));
        }
        fclose(pFile);
    }
    {
        //Label images
        FILE * pFile = fopen(
            ROOT_FOLDER"/data/t10k-labels-idx1-ubyte",
            "rb"
        );

        unsigned char inp[4];
        int num_read;
        num_read = fread(inp, 1, 4, pFile);
        int magic = reverseInt(inp);
        num_read = fread(inp, 1, 4, pFile);
        int num_labels = reverseInt(inp);
        assert(num_labels == imgs.size() && magic == 2048);
        unsigned char labels[num_labels];
        fread(labels, 1, num_labels, pFile);

        for(int n = 0;n<num_labels;n++){
            res.push_back(pair<Matrix<int>, int>(imgs[n], (int) labels[n]));
        }
    }

    return res;
}