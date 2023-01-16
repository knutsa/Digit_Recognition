#include "utils.hpp"
#include <stdio.h>

using namespace std;


void tool(){
    cout << "This was the first non-inline util implemented. It is useless!!!" << endl;
}

int reverseInt(unsigned char *a)
{
	return ((((a[0] * 256) + a[1]) * 256) + a[2]) * 256 + a[3];
}

//Return vector of images with corresponding labels, digits 0-9
datalist read_training_batch(int batch_size){
    datalist res;
    vector<Matrix<int> > imgs;

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

        assertm(magic == 2051 && h == 28 && w == 28 && num_imgs == 60000, "Error while reading data check that all paths are correct in utils.hpp.");

        for(int n = 0;n<num_imgs;n++){
            unsigned char img[28][28];
            fread(img, 28, 28, pFile);
            vector<vector<int> > data(28, vector<int>(28));
            for(int i = 0;i<28;i++){
                for(int j = 0;j<28;j++){
                    data[i][j] = (int) img[i][j];
                    assert(data[i][j] >= 0 && data[i][j] < 256);
                }
            }
            imgs.push_back(Matrix<int>(data));
        }
        assert(imgs.size() == 60000);
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
        assertm(num_labels == 60000 && magic == 2049, "Error while reading data check that paths are correct. Also in utils.hpp");
        unsigned char labels[num_labels];
        fread(labels, 1, num_labels, pFile);


        for(int n = 0;n<batch_size;n++){
            assert(labels[n] <10 && labels[n] >= 0);
            res.push_back(pair<Matrix<int>, int>(imgs[n], (int) labels[n]));
        }
        fclose(pFile);
    }

    return res;
}

datalist sample_data(const datalist &data, int sample_size){
    datalist sampled(sample_size);
    set<int> indexes_used;

    for(int i = 0;i<sample_size;i++){
        int index = i; // random() % data.size();
        while(indexes_used.find(index) != indexes_used.end()){
            index = (index+1) % data.size();
        }
        sampled[i] = data[index];
        indexes_used.insert(index);
    }

    return sampled;
}

//Return vector of images with corresponding labels, digits 0-9
datalist read_test_data(){
    datalist res;
    vector<Matrix<int> > imgs;

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

        assert(magic == 2051 && h == 28 && w == 28 && num_imgs == 10000);

        for(int n = 0;n<num_imgs;n++){
            unsigned char img[28][28];
            fread(img, 28, 28, pFile);
            vector<vector<int> > data(28, vector<int>(28));
            for(int i = 0;i<28;i++){
                for(int j = 0;j<28;j++){
                    data[i][j] = (int) img[i][j];
                    if (data[i][j] < 0 || data[i][j] >= 256)
                        cout << "Weirdo here " << data[i][j] << endl;
                    assert(data[i][j] >= 0 && data[i][j] < 256);
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
        assert(num_labels == 10000 && magic == 2049);

        unsigned char labels[num_labels];
        fread(labels, 1, num_labels, pFile);

        for(int n = 0;n<num_labels;n++){
            assert(labels[n] >= 0 && labels[n] < 10);
            res.push_back(pair<Matrix<int>, int>(imgs[n], (int) labels[n]));
        }
        fclose(pFile);
    }

    return res;
}

Matrix<int> max_pooling(Matrix<int> img, pair<int, int> pool_size = { 2,2 }) {
    int dy = pool_size.first, dx = pool_size.second;
    vector<vector<int> > pooled;
    for (int y = 0; y < img.h; y+=dy) {
        vector<int> pool_row;
        for (int x = 0; x < img.w; x += dx) {
            int M = INT16_MIN;
            for (int ky = 0; ky < dy && y+ky<img.h; ky++) {
                for (int kx = 0; kx < dx && x +kx<img.w; kx++) {
                    M = max(M, img.elements[y + ky][x + kx]);
                }
            }
            pool_row.push_back(M);
        }
        pooled.push_back(pool_row);
    }
    return Matrix<int>(pooled);
}

datalist preprocess(datalist data) {
    /*Hard coded pooling and convolution to process images before they are passed to the neural network.*/
    datalist res;
    for (auto data_point : data) {
        auto mod_img = max_pooling(data_point.first, { 2,2 });

        res.push_back({ mod_img, data_point.second });
    }
    return res;
}