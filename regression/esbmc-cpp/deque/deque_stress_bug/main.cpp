#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main() {
    deque<int> mydeque;
    const int SIZE = 20;
    
    // Fill with values 1 to SIZE
    for (int i = 1; i <= SIZE; i++) {
        mydeque.push_back(i);
    }
    
    // Reverse iterate and verify values
    int expected = SIZE;
    for (auto rit = mydeque.rbegin(); rit != mydeque.rend(); ++rit) {
        assert(*rit == expected);
        expected--;
    }
    
    assert(expected == 1); // it should fail     
    
    return 0;
}
