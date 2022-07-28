#include <iostream>
#include <string>

#define ROOT_FOLDER "/apa/"

using namespace std;

int main(){
    char table [2][3] = {{0,1,2}, {3,4,5}};

    cout << (int) table[0][1] << endl;
    cout << (void *) table[0] << endl;
    cout << (void *) table[1] << endl;
    cout << table << endl;


    string test = "hej "
    " dar "
    
    " på"    
    
    
    " dig";
    cout << test << endl;

    cout << sizeof(char) << endl;
    cout << sizeof(unsigned char) << endl;

    cout << ROOT_FOLDER"ej/dar" << endl;
    cout << "hej""dar" << endl;

}