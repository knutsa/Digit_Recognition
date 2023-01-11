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
        auto start = chrono::high_resolution_clock::now();
        this->epoch(data, batch_size);
        auto finished = chrono::high_resolution_clock::now();
        double time = chrono::duration_cast<chrono::nanoseconds>(finished - start).count() * 1e-9;
        cout << (epoch+1) << ":th epoch performed. (" << time << "s)" << ". Cost now is : " << endl;
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

#pragma omp parallel for
        for(int i = 0;i<to_process;++i){
            int data_index = processed + i;
            const int label = data[data_index].second;
            const auto img = data[data_index].first;
            vector<Matrix<double> > neurons_activation = this->forward_prop(img);
            auto probs = softmax(*(neurons_activation.end() - 1));
        
            back_prop(label, grad, neurons_activation, this->layers, probs );
        } 

        for(int l_ind = 0;l_ind < grad.size(); l_ind++){
            grad[l_ind] = grad[l_ind] * (1 / (double) to_process);

            for(int i = 0;i<grad[l_ind].h;i++){
                for(int j = 0;j<grad[l_ind].w-1;j++){
                    this->layers[l_ind].weights.elements[i][j] -= this->learning_rate * grad[l_ind].elements[i][j];
                }
                this->layers[l_ind].biases.elements[i][0] -= this->learning_rate * grad[l_ind].elements[i][grad[l_ind].w - 1];
            }
        }

        if (processed % 1000 == 0)
            cout << "\tProcessed: " << ((double)processed / (double)data.size() * 100) << " % of datapoints: " << processed << " grad[0] norm is: " << norm_square(grad[0]) << endl;
        processed += to_process;
    }

}

pair<double, double> DigitNetwork::cost_function(const datalist &data){
    long double cost = 0;
    int correct = 0, total = 0;
#pragma omp parallel for
    for (int data_index = 0;data_index<data.size();data_index++) {
        const int label = data[data_index].second;
        auto& img = data[data_index].first;
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