#include <math.h>
#include <algorithm>
#include <iomanip>
#include "network.hpp"

double norm_square(const Matrix<double>& A) {
    double normsq = 0;
    for (int i = 0; i < A.h; i++) {
        for (int j = 0; j < A.w; j++) {
            normsq += A.elements[i][j] * A.elements[i][j];
        }
    }
    return normsq;
}
double matrix_mean(const Matrix<double>& A) {
    double mean = 0;
    for (int i = 0; i < A.h;i++) {
        for (int j = 0; j < A.w; j++) {
            mean += A.elements[i][j];
        }
    }
    return mean / (A.h * A.w);
}

void DigitNetwork::train(const datalist& data, int epochs, int batch_size) {
    cout << "Training AI for " << epochs << " epochs with a batch size of " << batch_size << " and learning_rate = " << this->learning_rate << endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        this->epoch(data, batch_size);
        cout << (epoch+1) << " epochs performed. Cost now is:" << endl;
        auto performance = this->cost_function(data);
        cout << fixed << setprecision(10) << performance.first << " accuracy: " << performance.second << " %" << endl;
#ifdef DISPLAY
        for (int l_ind = 0; l_ind < this->layers.size(); l_ind++) {
            cout << "Data from layer" << l_ind << endl;
            cout << "\tnormsq of weights is " << norm_square(this->layers[l_ind].weights) << endl;
            cout << "\tMean of biases is " << matrix_mean(this->layers[l_ind].biases) << endl;
        }
        for (int i = 0; i < 3; i++) {
            auto neurons_activation = this->forward_prop(data[i].first);
            auto probs = this->analyze(data[i].first);
            cout << "Output neurons from image " << i << endl;
            (neurons_activation.end() - 1)->print(true);
            cout << "Output probability from image " << i << endl;
            probs.print(true);
        }

#endif // DISPLAY

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
        vector<Matrix<double> > grad;
        for(auto layer : this->layers){ grad.push_back(Matrix<double>(vector<vector<double> >(layer.weights.h, vector<double>(layer.weights.w+1)))); }

        for (int data_index = processed; data_index < processed+to_process;data_index++){

            const int label = data[data_index].second;
            const Matrix<int> img = data[data_index].first;
            vector<Matrix<double> > neurons_activation = this->forward_prop(img);
            auto probs = softmax(*(neurons_activation.end() - 1));
#ifdef DEBUG
            cout << "From epoch neurons activation is" << endl;
            for (auto nr : neurons_activation) {
                nr.print(true);
            }
#endif // DEBUG
        
            back_prop(label, grad, neurons_activation, this->layers, probs );
        } 

        for(int l_ind = 0;l_ind < grad.size(); l_ind++){
            grad[l_ind] = grad[l_ind] * (1 / (double) to_process);
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
                this->layers[l_ind].biases.elements[i][0] -= this->learning_rate * grad[l_ind].elements[i][grad[l_ind].w - 1];
            }
        }

        if (processed % 300 == 0)
            cout << "\tProcessed: " << ((double)processed / (double)data.size() * 100) << " % of datapoints: " << processed << " grad[0] norm is: " << norm_square(grad[0]) << endl;
        processed += to_process;
    }

}

pair<double, double> DigitNetwork::cost_function(const datalist &data){
    long double cost = 0;
    int correct = 0, total = 0;
    for(auto it : data){
        const int label = it.second;
        Matrix<int> &img = it.first;
        auto probs = this->analyze(img);
        int num_categories = probs.h; // number of output neurons from Network
        assert(label >= 0 && label < num_categories);
        double p = probs(label);
        assert(p > 0 && p <= 1);
        
        cost += -log(probs(label));
        total++;
        bool is_correct = true;
        for (int i = 0; i < probs.h; i++) {
            if (probs(i) >= p && i != label)
                is_correct = false;
        }
        if (is_correct)
            correct++; 
    }
    cost /= data.size();
    double accuracy = (double)correct / (double)total * 100;
    return make_pair(cost, accuracy);
}