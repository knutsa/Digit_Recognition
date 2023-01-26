#include "utils.hpp"
#include "network.hpp"

using namespace std;

void make_model() {

    cout << "Reading data" << endl;
    datalist full_data_set = read_training_batch();
    cout << "Done reading " << full_data_set.size() << " images" << endl;

    //full_data_set = preprocess(full_data_set);
    int hidden1 = 60, hidden2 = 60;
    DigitNetwork AI({ 784, hidden1, hidden2, 10 }, 2.0);
    cout << "Neural Network initialized with random weights. Network size is 784 x " << hidden1 << " x " << hidden2 << " x 10" << endl;
    cout << "Initial cost: ";
    auto res = AI.cost_function(full_data_set);
    double initial_cost = res.first, initial_accuracy = res.second;
    cout << fixed << setprecision(10) << initial_cost << " initial accuracy: " << initial_accuracy << "%" << endl;

    //Training
    AI.train(full_data_set, 15, 100);
    //AI.scale_learning(.75);
    //AI.train(full_data_set, 15, 100);
    //AI.scale_learning(0.75);
    //AI.train(full_data_set, 20, 100);

    cout << "AI is trained." << endl;

    cout << string(40, '=') << endl;
    cout << "Performance summary: " << endl;

    cout << "Reminding, initial cost was: " << initial_cost << " and initial accuracy was " << initial_accuracy << " %" << endl;
    auto final_res = AI.cost_function(full_data_set);
    double final_cost = final_res.first, final_accuracy = final_res.second;
    cout << "Final Cost is: " << final_cost << " and the network has a training accuracy of: " << final_accuracy << " %" << endl;

    datalist test_data = read_test_data();
    auto test_res = AI.cost_function(test_data);
    cout << "Test cost is: " << test_res.first << " and test accuracy is: " << test_res.second << "%" << endl;

    cout << "If you would like to save this model type in a name for the file in which to store it ('s' for saved_model) else press (ctr+C) to abort" << endl;
    string fn;
    cin >> fn;
    if (fn.size()) {
        if (fn == "s")
    fn = "saved_model";
    store_model(AI, test_res.first, test_res.second, fn);
    }
}

void run_saved_model() {
    DigitNetwork AI;
    string fn;

    cout << "Enter filename (without extension) where the model is stored, 's' for 'saved_model'" << endl;
    cin >> fn;
    while (true) {
        if (fn == "s")
            fn = "saved_model";
        try {
            AI = load_model(fn);
            break;
        }
        catch (const exception& e) {
            cout << "Couldn't read from '" << fn << "' try again with a different name" << endl;
        }
        cin >> fn;
    }
    cout << "AI loaded " << endl;
    
    auto testdata = read_test_data();
    auto stats = AI.cost_function(testdata);

    cout << "Test cost: " << stats.first << " accuracy: " << stats.second << "%" << endl;
}

int main() {
    cout << "Would you like to (1) train a new Nework, or (2) run a previously saved Network?" << endl;
    string user_input;
    while (true) {
        cin >> user_input;
        if (user_input == "1" || user_input == "2")
            break;
        cout << "Invalid option enter '1' or '2'" << endl;
    }
    if (user_input == "1") {
        int crashes = 0;
        while (true) {
            try {
                make_model();
                break;
            }
            catch (const exception& e) {
                cout << "Crashed " << crashes << " times" << endl;
                cout << e.what() << endl;
            }
        }
    }
    else {
        run_saved_model();
    }
}