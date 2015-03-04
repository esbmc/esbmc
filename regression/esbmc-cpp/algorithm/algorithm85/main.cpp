// replace_copy_if example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool IsOdd (int i) { return ((i%2)==1); }

int main () {
  vector<int> first,second;
  vector<int>::iterator it;

  // set some values:
  for (int i=1; i<5; i++) first.push_back(i);          // 1 2 3 4

  second.resize(first.size());   // allocate space
  replace_copy_if (first.begin(), first.end(), second.begin(), IsOdd, 0);
                                                        // 0 2 0 4
  assert(second[2] != 0);
  cout << "second contains:";
  for (it=second.begin(); it!=second.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}
