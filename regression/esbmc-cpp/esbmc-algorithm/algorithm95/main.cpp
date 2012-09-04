// rotate_copy algorithm example
#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main () {
  int myints[] = {10,20,30,40,50,60,70};

  vector<int> myvector;
  vector<int>::iterator it;

  myvector.resize(7);

  rotate_copy(myints,myints+3,myints+7,myvector.begin());
  assert(myvector[1] != 50);
  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
