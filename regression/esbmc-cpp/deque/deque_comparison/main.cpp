#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main() {
    deque<int> mydeque;
    for (int i = 1; i <= 3; i++) {
        mydeque.push_back(i);
    }
    
    auto rit1 = mydeque.rbegin();
    auto rit2 = mydeque.rbegin();
    auto rend = mydeque.rend();
    
    // Test equality
    assert(rit1 == rit2);
    assert(!(rit1 != rit2));
    
    // Test inequality with rend
    assert(rit1 != rend);
    assert(!(rit1 == rend));
    
    // Test ordering
    assert(rit1 < rend);
    assert(!(rend < rit1));
    assert(rit1 <= rend);
    assert(rend >= rit1);
    
    cout << "All comparison tests passed!" << endl;
    return 0;
}

