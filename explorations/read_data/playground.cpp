#include <iostream>
#include <string>
#include <vector>

#define ROOT_FOLDER "/apa/"

using namespace std;


struct Car {
    int age;
    int price;
    float weight;
};


void increase(int &x){
    x++;
}

class MyClass{
public:
    Car car1;
    vector<vector<int> > data;

    MyClass(){}

    void f(){
        Car car1;
    }

    int operator[](pair<int, int> ind){
        int i = ind.first, j = ind.second;

        return 10*i + j;
    }
};

int main(){

    MyClass A;
    MyClass &B = A;
    A.data = {{1,2,3}, {4,5,6}};
    cout << B.data[0][0] << endl;
    B.data[0][0] -= 100;
    cout << A.data[0][0] << endl;

    int x = 10;
    increase(x);
    cout << x << endl;

}