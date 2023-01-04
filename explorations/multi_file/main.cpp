#include <iostream>

using namespace std;

int add(int x, int y);
void test_ref(int& x);
int add_one(int x);
int no_modifier(const int& x);

int main(){

    int x = 10, y = 20;
    test_ref(x);
    cout << x << endl;
    int z = no_modifier(y);
    cout << y << " z = " << z << endl;

    return 0;
}