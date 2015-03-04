// remove_if example
#include <iostream>
#include <algorithm>
#include <cassert>
using namespace std;

bool IsOdd (int i) { return ((i%2)==1); }

int main () {
  int myints[] = {1,2,3,4,5,6,7,8,9};            // 1 2 3 4 5 6 7 8 9

  // bounds of range:
  int* pbegin = myints;                          // ^
  int* pend = myints+sizeof(myints)/sizeof(int); // ^                 ^

  pend = remove_if (pbegin, pend, IsOdd);        // 2 4 6 8 ? ? ? ? ?
                                                 // ^       ^
  assert(myints[0] == 2);
  assert(myints[1] == 4);
  assert(myints[2] == 6);
  assert(myints[3] == 8);
  cout << "range contains:";
  for (int* p=pbegin; p!=pend; ++p)
    cout << " " << *p;

  cout << endl;
 
  return 0;
}
