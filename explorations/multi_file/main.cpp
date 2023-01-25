#include <iostream>
#include <iomanip>
#include <random>
#include <map>
#include <algorithm>

using namespace std;

int add(int x, int y);
void test_ref(int& x);
int add_one(int x);
int no_modifier(const int& x);

int main(){

    random_device rd;

    default_random_engine generator(time(0));
    normal_distribution<double> distribution(5, 2);

    vector<int> data = { 0,1,2,3,4,5,6,7,8,9 };

    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);

    for (auto x : data)
        cout << x << ", ";
    cout << endl;

    cout << "5 random X from Normal distribution" << endl;
    for (int i = 0; i < 5; i++)
        cout << distribution(generator) << endl;

    /*map<int, int> hist;
    for (int i = 0; i < 10000; i++) {
        ++hist[round(distribution(generator))];
    }
    for (auto p : hist) {
        std::cout << std::setw(2)
            << p.first << ' ' << std::string(p.second / 50, '*') << '\n';
    }*/

    return 0;
}