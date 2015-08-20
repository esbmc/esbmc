// replace_if example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool IsOdd (int i) { return ((i%2)==1); }

int main () {
  vector<int> myvector;
  vector<int>::iterator it;

  // set some values:
  for (int i=1; i<6; i++) myvector.push_back(i);          // 1 2 3 4 5 
  
  assert(myvector[2] == 3);

  replace_if (myvector.begin(), myvector.end(), IsOdd, 0); // 0 2 0 4 0

  assert(myvector[2] == 3);
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}
