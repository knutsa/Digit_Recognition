#include <math.h>
#include "network.hpp"

void back_prop(const int &label,vector<Matrix<double> > &grad, vector<Matrix<double> > &neurons_activation, const vector<Layer> &layers){
    //Adds contribution of img to grad from img. Grad is in format list[weight matrix | biases]
    // cout << "ENTERING BACK PROP" << endl << endl; 
    vector<double> calced(10, 0);
    assert(neurons_activation.size() == grad.size()+1); //one layer between each layer of neurons
    Matrix<double> &result = neurons_activation[neurons_activation.size()-1];

    // cout << "result" << endl;
    for(int i = 0;i<result.h;i++){
        // cout << result[i] << ' ';
    }
    // cout << endl;
    // cout << "Calced"<<endl;
    for(int i = 0;i<10;i++){
        calced[i] = 2* ( (i == label) ? result[i]-1 : result[i] );
        // cout << calced[i]<<' ';
    }
    // cout << endl;
    // cout << "Neuron activation " <<neurons_activation.size() << endl;
    for(auto act : neurons_activation){
        for(int i = 0;i<min((int) act.h, 784);i++){
            // cout << act[i] << ' ';
        }
        // cout << endl;
    }
    
    for(int layer_index = grad.size()-1;layer_index>=0;layer_index--){
        Matrix<double> &mat = grad[layer_index];
        assert(mat.h == calced.size());
        vector<double> new_calced(mat.w-1);
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
    // for(int gri = 0;gri<grad.size();gri++){
    //     auto gr = grad[gri];
    //     cout << "From Grad back-prop " << gr.h << ' ' << gr.w << endl;
    //     gr.print(gri == 2);
    // }
}

void DigitNetwork::train(const datalist &data){

    vector<Matrix<double> > grad;

    for(auto layer : this->layers){
        grad.push_back(Matrix<double>(vector<vector<double> >(layer.weights.h, vector<double>(layer.weights.w+1))));
    }

    int counter = 0;
    // cout << data.size() << endl;
    for(auto it : data){
        counter++;
        const int label = it.second;
        const Matrix<int> &img = it.first;
        vector<Matrix<double> > neurons_activation = this->analyze(img);
        
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

double norm_square(const Matrix<double> &A){
    double normsq = 0;
    for(int i = 0;i<A.h;i++){
        for(int j = 0;j<A.w;j++){
            normsq += A.elements[i][j]*A.elements[i][j];
        }
    }
    return normsq;
}

double DigitNetwork::cost_function(const datalist &data){
    double res = 0;
    for(auto it : data){
        const int label = it.second;
        Matrix<int> &img = it.first;
        vector<Matrix<double> > neurons_activation = this->analyze(img);

        Matrix<double> correct({0,0,0,0,0,0,0,0,0,0}); //10 0s
        correct.elements[label][0] = 1;
        res += norm_square(correct - neurons_activation[neurons_activation.size()-1]);
    }
    
    return sqrt(res / data.size());
}