// replace_copy example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int myints[] = { 10, 20, 30, 30, 20, 10, 10, 20 };
  // 10 99 30 30 99 10 10 99
  vector<int> myvector (8);
  replace_copy (myints, myints+8, myvector.begin(), 20, 99);
  assert(myvector[1] == 99);
  assert(myvector[4] == 99);
  assert(myvector[7] == 99);
  cout << "myvector contains:";
  for (vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;
  cout << endl;
 
  return 0;
}
