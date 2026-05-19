// unique_copy example
//
// Note: vector::assign() invalidates all iterators per the C++ standard
// ([sequence.reqmts] / [vector.modifiers]).  The original cplusplus.com
// example here reused a pre-assign iterator across an assign() call, which
// is undefined behaviour.  The test was marked KNOWNBUG because ESBMC
// (correctly) flagged the resulting dangling deref as an array-bounds
// violation.  Updated to recompute the iterator after assign so the test
// exercises unique_copy(..., pred) on a well-defined input range.
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i, int j) {
  return (i==j);
}

int main () {
  int myints[] = {10,20,20,20,30,30,20,20,10};
  int myints1[] = {10,10,20,20,30,0,0,0,0};
  vector<int> myvector (9);
  vector<int>::iterator it;

  // using default comparison: 5 unique consecutive values written.
  it = unique_copy (myints, myints+9, myvector.begin());   // 10 20 30 20 10
  assert(it - myvector.begin() == 5);

  // assign() invalidates iterators; recompute the input range below.
  myvector.assign(myints1, myints1+9);                     // 10 10 20 20 30 0 0 0 0

  // using predicate comparison on the first 5 elements: {10,10,20,20,30}
  // yields {10,20,30}; positions 3 and 4 are left untouched.
  it = unique_copy (myvector.begin(), myvector.begin()+5,
                    myvector.begin(), myfunction);         // 10 20 30 20 30
  assert(it - myvector.begin() == 3);

  assert(myvector[0] == 10);
  assert(myvector[1] == 20);
  assert(myvector[2] == 30);
  assert(myvector[3] == 20);
  assert(myvector[4] == 30);

  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
