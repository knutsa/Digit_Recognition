#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
	fstream f;
	f.open("write2.txt");
	vector<string> strs;
	string inp;
	getline(f, inp);
	vector<double> x;
	size_t pos;
	double xi;
	string  str;
	for (int i = 0; i < inp.size(); i++) {
		if (inp[i] == ',') {
			x.push_back(stod(str));
			str = "";
		}
		else {
			str += inp[i];
		}
	}
	if (str.size())
		x.push_back(stod(str));
	for (auto a : x)
		cout << a << endl;
	//f >> str;
	cout << "Done" << endl;
	int a = 10;
	char b = (char)a;
	cout << (b == 10) << endl;
}