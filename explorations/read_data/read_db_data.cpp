#include <iostream>
#include "../../src/utils.hpp"

using namespace std;

int reverseInt(unsigned char *a)
{
	return ((((a[0] * 256) + a[1]) * 256) + a[2]) * 256 + a[3];
}

int main(){
    //THIS CODE WORKS
    FILE * pFile = fopen("../../data/train-images-idx3-ubyte", "rb");

    if(pFile == NULL){
        cout << "Could not open file" << endl;
        return 1;
    }

    unsigned char a[4];
    int num_read;
    num_read = fread(a, 1, 4, pFile);
    const int magic = reverseInt(a);

    cout << magic << endl;
    fread(a, 1, 4, pFile);
    const int num_imgs = reverseInt(a);

    fread(a, 1, 4, pFile);
    const int num_rows = reverseInt(a);

    fread(a, 1, 4, pFile);
    const int num_cols = reverseInt(a);

    cout << num_read << endl;
    cout << "Data is of dim: " << num_imgs << " x " << num_rows << " x " << num_cols << endl;

    vector<Matrix<int> > imgs;
    for(int img_id = 0;img_id<num_imgs;img_id++){
        unsigned char img[num_rows][num_cols];

        fread(img, num_cols, num_rows, pFile);
        vector<vector<int> > data(28, vector<int>(28));
        for(int i = 0;i<28;i++){
            for(int j = 0;j<28;j++){
                data[i][j] = (int) img[i][j];
            }
        }
        imgs.push_back(Matrix<int>(data));
    }
    cout << "All image data read" << endl;
    fclose(pFile);

    FILE * pFileLabels = fopen("../../data/train-labels-idx1-ubyte", "rb");
    num_read = fread(a, 1, 4, pFileLabels);
    int magicLabels = reverseInt(a);

    num_read = fread(a, 1, 4, pFileLabels);
    int num_labels = reverseInt(a);

    cout << "Label data" << endl;
    cout << magicLabels << endl;
    cout << "Data is of dimension: " << num_labels << endl;


    assert(num_labels == num_imgs);
    vector<pair<Matrix<int>, int > > data;
    unsigned char labels[num_labels];
    fread(labels, 1, num_labels, pFileLabels);
    cout << (int) labels[0] << ' ' << (int) labels[1] << endl;
    for(int n = 0;n<num_labels;n++){
        data.push_back(pair<Matrix<int>, int>(imgs[n], (int) labels[n]));
    }

    return 0;
    //THIS CODE WORKS
}