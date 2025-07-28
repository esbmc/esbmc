#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main() {
    deque<int> mydeque;
    for (int i = 1; i <= 4; i++) {
        mydeque.push_back(i);
    }
    
    auto rit1 = mydeque.rbegin();
    auto rit2 = mydeque.rbegin();
    
    // Test pre-increment
    int val1 = *rit1;  // Should be 4
    ++rit1;
    int val2 = *rit1;  // Should be 3
    
    // Test post-increment
    int val3 = *(rit2++);  // Should be 4, then rit2 moves
    int val4 = *rit2;      // Should be 3
    
    assert(val1 == 4);
    assert(val2 == 3);
    assert(val3 == 4);
    assert(val4 == 3);
    assert(rit1 == rit2);  // Both should point to same position
    
    cout << "Increment tests passed!" << endl;
    return 0;
}

