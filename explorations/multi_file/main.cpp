#include <iostream>
#include <iomanip>
#include <random>
#include <map>

using namespace std;

int add(int x, int y);
void test_ref(int& x);
int add_one(int x);
int no_modifier(const int& x);

int main(){

    default_random_engine generator;
    normal_distribution<double> distribution(5, 2);

    map<int, int> hist;
    for (int i = 0; i < 10000; i++) {
        ++hist[round(distribution(generator))];
    }
    for (auto p : hist) {
        std::cout << std::setw(2)
            << p.first << ' ' << std::string(p.second / 50, '*') << '\n';
    }

    return 0;
}