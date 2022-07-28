#include <iostream>
#include <fstream>

using namespace std;

int main(){
    cout << "Testing file reading"<<endl;

    ifstream indata;
    indata.open("data.txt");
    if(!indata) {
        cout << "Could not open file!"<<endl;
        return 1;
    }     

    int val;
    while(!indata.eof()){
        indata >> val;
        cout << val << endl;
    }

    cout << "Done"<<endl;

    return 0;
}