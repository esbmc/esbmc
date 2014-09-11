// replace algorithm example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int myints[] = { 10, 20, 30 };
  vector<int> myvector (myints, myints+3);            // 10 20 30 

  replace (myvector.begin(), myvector.end(), 20, 99); // 10 99 30 

  assert(myvector[1] != 99);
  cout << "myvector contains:";
  for (vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}
