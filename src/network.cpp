#include <math.h>
#include <algorithm>
#include <iomanip>
#include "network.hpp"

void back_prop(const int label,vector<Matrix<double> > &grad,const vector<Matrix<double> > &neurons_activation, const vector<Layer> &layers){
    /*Adds contribution of img to grad from img. Grad is a Matrix of the format [weight matrix | biases column]*/
    assert(neurons_activation.size() == grad.size()+1); //one layer between each group of neurons
    Matrix<double> result = neurons_activation[neurons_activation.size()-1]; 
    vector<double> calced(result.h, 0); //Derivative with respect to output neurons

    //Derivative of cost function with respect to outpput neurons 
    for(int i = 0;i<result.h;i++){
        calced[i] = 2* ( (i == label) ? result[i]-1 : result[i] );
    }
#ifdef DEBUG
    cout << "Derivative with respect to output neurons" << endl;
    for (auto it : calced) { cout << it << endl; }
    /*cout << "Output Neurons" << endl;
    Matrix<double> copy = *(neurons_activation.end() - 1);
    copy.print(true);*/
#endif // DEBUG
    
    for(int layer_index = grad.size()-1;layer_index>=0;layer_index--){ //process layer of weights in reverse order
        Matrix<double> &mat = grad[layer_index];
        const Layer &current_layer = layers[layer_index];
        assert(mat.h == current_layer.weights.h && mat.w == current_layer.weights.w + 1);
        assert(mat.h == calced.size());
        vector<double> new_calced(mat.w-1);

        for(int i = 0;i<mat.h;i++){
            for(int j = 0;j<mat.w;j++){ //last one is the bias
                if(j == mat.w-1){ 
                    mat.elements[i][j] += sigmoid_derivative(calced[i]); //derivative with respect to bias
                } else { 
                    mat.elements[i][j] += neurons_activation[layer_index].elements[j][0] * sigmoid_derivative(calced[i]); //Derivative with respect to weight
                    new_calced[j] += current_layer.weights.elements[i][j] * sigmoid_derivative(calced[i]); //Derivative with respect to neuron value --- only used to propagate backwards
                }
            }
        }
        calced = new_calced;
    }
#ifdef DEBUG
    cout << "Derivative with respect to input neurons" << endl;
    for (auto it : calced) { cout << it << endl; }
#endif // DEBUG

}

void DigitNetwork::train(const datalist& data, int epochs, int batch_size) {
    cout << "Training AI for " << epochs << " epochs with a batch size of " << batch_size << endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        this->epoch(data, batch_size);
        cout << (epoch+1) << " epochs performed. Cost now is:" << endl;
        cout << fixed << setprecision(10) << this->cost_function(data) << endl;
    }
#ifdef DEBUG
    for (auto layer : this->layers) {
        cout << "Layer weights" << endl;
        layer.weights.print();
        cout << "Layer biases" << endl;
        layer.biases.print();
    }
#endif // DEBUG

}

void DigitNetwork::epoch(datalist data, int batch_size){
    /*Process every data point once to fit model*/
    //Shuffle data for random batches
    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);

    int processed = 0;
    while(processed < data.size()) {
        int to_process = min(batch_size, ((int) data.size()) - processed);
        if(processed % 1000 == 0)
            cout << "\tProcessed: " << ((double) processed / (double) data.size() * 100) << " % of datapoints: " << processed << endl;
        
        vector<Matrix<double> > grad;
        for(auto layer : this->layers){ grad.push_back(Matrix<double>(vector<vector<double> >(layer.weights.h, vector<double>(layer.weights.w+1)))); }

        for (int data_index = processed; data_index < processed+to_process;data_index++){

            const int label = data[data_index].second;
            const Matrix<int> img = data[data_index].first;
            vector<Matrix<double> > neurons_activation = this->analyze(img);
#ifdef DEBUG
            cout << "From epoch neurons activation is" << endl;
            for (auto nr : neurons_activation) {
                nr.print(true);
            }
#endif // DEBUG

        
            back_prop(label, grad, neurons_activation, this->layers );
        }
 

        for(int l_ind = 0;l_ind < grad.size(); l_ind++){
            grad[l_ind] = grad[l_ind] * ( 1 / (double) to_process);
#ifdef DEBUG
            for (int l_ind = 0; l_ind < grad.size(); l_ind++) {
                cout << "Learning rate is " << this->learning_rate << " and after recent backprop " << to_process << " datapoints this is the gradiend with respect to the weights of layer: " << l_ind << endl;
                grad[l_ind].print(true);
            }
#endif // DEBUG

#ifdef DEBUG
            cout << "Layer " << l_ind << " before GD : " << endl;
            cout << "Weights" << endl;
            this->layers[0].weights.print(true);
            cout << "Biases" << endl;
            this->layers[0].biases.print(true);
#endif // DEBUG
            for(int i = 0;i<grad[l_ind].h;i++){
                for(int j = 0;j<grad[l_ind].w-1;j++){
                    this->layers[l_ind].weights.elements[i][j] -= this->learning_rate * grad[l_ind].elements[i][j];
                }
                this->layers[l_ind].biases.elements[i][0] -= this->learning_rate * grad[l_ind].elements[i][grad[l_ind].w-1];
            }
        }
        processed += to_process;
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
        int num_categories = (this->layers.end() - 1)->weights.h; // number of output neurons from Network
        assert(label >= 0 && label < num_categories);

        Matrix<double> correct(vector<double>(num_categories, 0));
        correct.elements[label][0] = 1;
        res += norm_square(correct - neurons_activation[neurons_activation.size()-1]);
    }
    
    return sqrt(res / data.size());
}