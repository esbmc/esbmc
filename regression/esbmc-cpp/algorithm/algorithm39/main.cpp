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
  
  assert(IsOdd(myvector[0]));
  assert(IsOdd(myvector[1]));
  assert(IsOdd(myvector[2]));
  assert(IsOdd(myvector[3]));
  assert(IsOdd(myvector[4]));
  assert(!IsOdd(myvector[5]));
  assert(!IsOdd(myvector[6]));
  assert(!IsOdd(myvector[7]));
  assert(!IsOdd(myvector[8]));
  

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
