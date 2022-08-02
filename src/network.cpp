#include <math.h>
#include "network.hpp"

#define sigmoid(x) 1 / (1 + exp(-x))
#define sigmoid_derivative(x) 1 / ( (exp(x/2) + exp(-x/2)) *(exp(x/2)+exp(-x/2)) )


Matrix<float> DigitNetwork::analyze(Matrix<int> img){
    vector<float> img_row;
    for(int i = 0;i<img.h;i++){
        for(int j = 0;j<img.w;j++){
            img_row.push_back((float) img.elements[i][j]);
        }
    }
    Matrix<float> data(img_row);
    for(auto lr : this->layers){
        data = lr.weights * data + lr.biases;
        for(int i = 0;i<data.h;i++){
            data.elements[i][0] = sigmoid(data.elements[i][0]);
        }
    }

    return data;
}

void back_prop(const int &label,vector<Matrix<float> > &grad, const vector<vector<float> > &neurons_activation, const vector<Layer> &layers){
    //Adds contribution of img to grad from img. Grad is in format list[weight matrix | biases]
    cout << "ENTERING BACK PROP" << endl << endl; 
    vector<float> calced(10, 0);
    assert(neurons_activation.size() == grad.size()+1); //one layer between each layer of neurons
    const vector<float> &result = neurons_activation[neurons_activation.size()-1];

    cout << "Calced"<<endl;
    for(int i = 0;i<10;i++){
        calced[i] = 2* ( (i == label) ? result[i]-1 : result[i] );
        cout << calced[i]<<' ';
    }
    cout << endl;
    
    for(int layer_index = grad.size()-1;layer_index>=0;layer_index--){
        Matrix<float> &mat = grad[layer_index];
        assert(mat.h == calced.size());
        vector<float> new_calced(mat.w-1);
        for(int i = 0;i<mat.h;i++){
            for(int j = 0;j<mat.w;j++){ //last one is the bias
                if(j == mat.w-1){ //pad neurons vec with 1
                    mat.elements[i][j] += sigmoid_derivative(calced[i]);
                } else {
                    mat.elements[i][j] += neurons_activation[layer_index][j] * sigmoid_derivative(calced[i]);
                    new_calced[j] += layers[layer_index].weights.elements[i][j] * sigmoid_derivative(calced[i]);
                }
            }
        }
        calced = new_calced;
    }
    for(auto gr : grad){
        cout << "From Grad back-prop" << endl;
        gr.print();
    }
}

void DigitNetwork::train(const datalist &data){

    vector<Matrix<float> > grad;

    for(auto layer : this->layers){
        grad.push_back(Matrix<float>(vector<vector<float> >(layer.weights.h, vector<float>(layer.weights.w+1))));
    }

    int counter = 0;
    cout << data.size() << endl;
    for(auto it : data){
        // if(counter % 1000 == 0)
            cout << "trained " << counter << endl;
        counter++;
        vector<vector<float> > neurons_activation(this->layers.size()+1);
        const Matrix<int> &img = it.first;
        const int label = it.second;
        for(int i = 0;i<img.h;i++){
            for(int j = 0;j<img.w;j++){
                neurons_activation[0].push_back((float) img.elements[i][j] / 256.0 );
            }
        }

        Matrix<float> curr(neurons_activation[0]);
        for(int layer_index = 0;layer_index<this->layers.size();layer_index++){
            Layer &layer = this->layers[layer_index];
            curr = layer.weights*curr + layer.biases;
            for(int i = 0;i<curr.h;i++){
                curr.elements[i][0] = sigmoid(curr.elements[i][0]);
            }
            for(int i = 0;i<curr.h;i++){
                neurons_activation[layer_index+1].push_back(curr.elements[i][0]);
            }
        }
        

       back_prop(label, grad, neurons_activation, this->layers );

    }
    for(int l_ind = 0;l_ind < grad.size(); l_ind++){
        grad[l_ind] = grad[l_ind] * ( 1 /  data.size());
        
        for(int i = 0;i<grad[l_ind].h;i++){
            for(int j = 0;j<grad[l_ind].w-1;j++){
                this->layers[l_ind].weights.elements[i][j] -= this->learning_rate * grad[l_ind].elements[i][j];
            }
            this->layers[l_ind].biases.elements[i][0] -= this->learning_rate * grad[l_ind].elements[i][grad[l_ind].w-1];
        }
    }
}