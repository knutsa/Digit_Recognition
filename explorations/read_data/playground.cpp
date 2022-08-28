#include <iostream>
#include <string>
#include <vector>
#include <random>

#define ROOT_FOLDER "/apa/"

using namespace std;


struct Car {
    int age;
    int price;
    double weight;
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

    default_random_engine generator;
    normal_distribution<double> distribution(.0, 4.0);


    for(int i = 0;i<10;i++){
        double x = distribution(generator);

        cout << x << endl;
    }

}