#include <fstream>
#include "utils.hpp"
#include "network.hpp"

/*Store model weighs and biases in st*/
void store_model(const DigitNetwork& AI, double loss, double accuracy, string filename) {

	ofstream storage;
	storage.open(ROOT_FOLDER"\\"+filename+".csv");
	storage << "Stored Network parameters. " << AI.loss_rep() << " Loss: " <<  loss << " Accuracy : " << accuracy << " % \n";
	storage << (int) AI.get_loss() << "\n";

	for (int layer_index = 0; layer_index < AI.layers.size(); layer_index++) {
		const auto& layer = AI.layers[layer_index];
		size_t h = layer.weights.h, w = layer.weights.w;
		storage << "Layer " << layer_index << " of size " << h << " x " << (w+1) << "\n";
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				storage << layer.weights.elements[i][j] << ",";
			}
			storage << layer.biases.elements[i][0] << "\n";
		}
	}
	storage.close();
	cout << "Model stored in " << ROOT_FOLDER << "\\" << filename << ".csv" << endl;
}

vector<double> inline read_csv_floats(const string& line){
	string str;
	vector<double> res;
	
	for (int i = 0; i < line.size(); i++) {
		if (line[i] == ',') {
			assertm(str.size(), "Failed to read stored model");
			try { res.push_back(stod(str)); }
			catch(const exception& e) { assertm(1 == 0, "Failed to read stored model"); }
			str = "";
		}
		else
			str += line[i];
	}
	if (str.size()) {
		try { res.push_back(stod(str)); }
		catch(const exception& e) { assertm(1 == 0, "Failed to read stored model"); }
	}

	return res;
}

DigitNetwork load_model(string filename) {
	ifstream storage;
	storage.open(ROOT_FOLDER"/"+filename+".csv");

	string inp;
	getline(storage, inp);
	assertm(inp.substr(0, 26) == "Stored Network parameters.", "reading from ill formated or empty 'stored_model' file");
	getline(storage, inp);
	int loss_option = stoi(inp);
	assertm(loss_option == L2 || loss_option == CROSS_CATEGORICAL_ENTROPY, "invalid loss found in stored model");
	vector<Layer> layers;
	size_t curr_h, curr_w;
	vector<vector<double> > lines_read;
	int layer_index = 0;

	while (inp.size()) {

		getline(storage, inp);
		if (inp.substr(0, 5) == "Layer") {
			if(lines_read.size())
				layers.push_back(Layer(lines_read));
			layer_index++; 
			lines_read.clear();
			continue;
		}
		auto pars = read_csv_floats(inp);
		if (pars.size()) {
			if (lines_read.size()) {
				assertm((lines_read.end() - 1)->size() == pars.size(), "ill formated csv data");
			}
			lines_read.push_back(pars);
		}
	}
	if(lines_read.size())
		layers.push_back(Layer(lines_read));

	return DigitNetwork(layers, loss_option);
}