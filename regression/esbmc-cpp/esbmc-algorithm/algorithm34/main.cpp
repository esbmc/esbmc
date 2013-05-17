// reverse_copy example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int myints[] ={1,2,3,4,5,6,7,8,9};
  vector<int> myvector;
  vector<int>::iterator it;

  myvector.resize(9);

  reverse_copy (myints, myints+9, myvector.begin());
  assert(myvector[0] == 9);
  assert(myvector[1] == 8);
  assert(myvector[2] == 7);
  assert(myvector[3] == 6);
  assert(myvector[4] == 5);

  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
