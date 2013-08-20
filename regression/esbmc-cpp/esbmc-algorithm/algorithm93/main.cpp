// reverse_copy example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int myints[] ={1,2,3,4};
  vector<int> myvector;
  vector<int>::iterator it;

  myvector.resize(4);

  reverse_copy (myints, myints+4, myvector.begin());
  assert(myvector[0] != 4);
  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
