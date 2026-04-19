#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main() {
    // Test empty deque
    deque<int> empty_deque;
    assert(empty_deque.rbegin() == empty_deque.rend());
    
    // Test single element
    deque<int> single_deque;
    single_deque.push_back(42);
    
    auto rit = single_deque.rbegin();
    assert(*rit == 42);
    assert(rit != single_deque.rend());
    
    ++rit;
    assert(rit == single_deque.rend());
    
    cout << "Edge case tests passed!" << endl;
    return 0;
}

