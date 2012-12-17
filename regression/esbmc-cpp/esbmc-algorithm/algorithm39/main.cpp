// stable_partition example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool IsOdd (int i) { return (i%2)==1; }

int main () {
  vector<int> myvector;
  vector<int>::iterator it, bound;
  
//  odd members: 1 3 5 7 9
//even members: 2 4 6 8


  // set some values:
  for (int i=1; i<10; ++i) myvector.push_back(i); // 1 2 3 4 5 6 7 8 9

  bound = stable_partition (myvector.begin(), myvector.end(), IsOdd);
  
  assert(myvector[0] == 1);
  assert(myvector[1] == 3);
  assert(myvector[2] == 5);
  assert(myvector[3] == 7);
  assert(myvector[4] == 9);
  assert(myvector[5] == 2);
  assert(myvector[6] == 4);
  assert(myvector[7] == 6);
  assert(myvector[8] == 8);
  

  // print out content:
  cout << "odd members:";
  for (it=myvector.begin(); it!=bound; ++it)
    cout << " " << *it;

  cout << "\neven members:";
  for (it=bound; it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
