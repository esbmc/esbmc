#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main() {
    deque<int> mydeque;
    for (int i = 1; i <= 5; i++) {
        mydeque.push_back(i);
    }
    
    auto rit = mydeque.rbegin();
    
    // Test operator+
    auto rit2 = rit + 2;
    assert(*rit == 5);   // Original unchanged
    assert(*rit2 == 3);  // Moved 2 positions backward
    
    // Test operator-
    auto rit3 = rit2 - 1;
    assert(*rit3 == 4);  // Moved 1 position forward
    
    // Test operator+=
    rit += 3;
    assert(*rit == 2);   // Moved 3 positions backward
    
    // Test operator-=
    rit -= 1;
    assert(*rit == 3);   // Moved 1 position forward
    
    cout << "Arithmetic tests passed!" << endl;
    return 0;
}

