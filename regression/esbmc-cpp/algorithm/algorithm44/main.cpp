// nth_element example
//
// std::nth_element only guarantees that the element at the nth position is
// the one that would be there in a fully sorted sequence, with elements
// before it <= it and elements after it >= it.  It does NOT guarantee any
// specific permutation of the surrounding elements.  The original assertions
// here checked an implementation-specific permutation, which is why this
// test was marked KNOWNBUG.  Updated to assert only what the C++ standard
// guarantees ([alg.nth.element]).
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i,int j) { return (i<j); }

int main () {
  vector<int> myvector;
  vector<int>::iterator it;

  // set some values:
  for (int i=1; i<10; i++) myvector.push_back(i);   // 1 2 3 4 5 6 7 8 9

  // using default comparison (operator <):
  nth_element (myvector.begin(), myvector.begin()+5, myvector.end());

  // The element at position 5 must be the 6th-smallest, i.e. 6.
  assert(myvector[5] == 6);
  // Elements before the partition point are <= the pivot.
  for (int i = 0; i < 5; ++i)
    assert(myvector[i] <= myvector[5]);
  // Elements after the partition point are >= the pivot.
  for (int i = 6; i < 9; ++i)
    assert(myvector[i] >= myvector[5]);

  // using function as comp
  nth_element (myvector.begin(), myvector.begin()+5, myvector.end(), myfunction);
  assert(myvector[5] == 6);

  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
