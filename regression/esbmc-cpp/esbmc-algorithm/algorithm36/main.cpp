// rotate_copy algorithm example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int myints[] = {10,20,30,40,50,60,70};
// 40 50 60 70 10 20 30
  vector<int> myvector;
  vector<int>::iterator it;

  myvector.resize(7);

  rotate_copy(myints,myints+3,myints+7,myvector.begin());
  assert(myvector[0] == 40);
  assert(myvector[1] == 50);
  assert(myvector[2] == 60);
  assert(myvector[3] == 70);
  
    // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
