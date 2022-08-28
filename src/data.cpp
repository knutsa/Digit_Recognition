#include "utils.hpp"

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
        fclose(pFile);
    }
    cout << " images read." << endl;
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
        cout << "Labels read " << endl;

        for(int n = 0;n<batch_size;n++){
            assert(labels[n] <10 && labels[n] >= 0);
            res.push_back(pair<Matrix<int>, int>(imgs[n], (int) labels[n]));
            // res.push_back(DataPoint(imgs[n], labels[n]));
        }
    }
    cout << "Done " << "returning datalist of length " << batch_size <<endl;

    return res;
}

datalist sample_data(const datalist &data, int sample_size){
    datalist sampled(sample_size);
    set<int> indexes_used;

    for(int i = 0;i<sample_size;i++){
        int index = random() % data.size();
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

        assert(magic == 2051 && h == 28 && w == 28);

        for(int n = 0;n<num_imgs;n++){
            unsigned int img[28][28];
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
            // res.push_back(DataPoint(imgs[n], labels[n]));
        }
    }

    return res;
}