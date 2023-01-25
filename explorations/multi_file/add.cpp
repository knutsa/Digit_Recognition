
int add(int x, int y){
    return x + y;
}

void test_ref(int& x) {
    x++;
}

int add_one(int x) {
    x++;
    return x;
}

int no_modifier(const int &x) {
    return add_one(x);
}