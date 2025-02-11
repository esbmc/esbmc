#include <cassert>

int main() 
{
    int* p = new int(10);
    double* s = new double(9.8);
    bool* b = new bool(true);

    assert(*p == 0); //should be 10
    assert(*s == 0); //should be 9.8
    assert(*b == true);

    return 0;
}
