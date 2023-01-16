#include <iostream>
#include <fstream>

using namespace std;

int main() {
	
	fstream f;
	f.open("write2.txt");

	f << "Hello writing!";


	return 0;
}