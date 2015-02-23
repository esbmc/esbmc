#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  int b[4] = {3,6,9,12};
  multiset<int> myset(b,b+4);
  int i;

  // set some initial values:
//  for (i=1; i<5; i++) myset.insert(i*3);    // set: 3 6 9 12
  assert(myset.count(3) == 1);
  assert(myset.count(6) == 1);
  assert(myset.count(9) != 1);
  assert(myset.count(12) == 1);
  myset.insert(3);
  myset.insert(3);
  myset.insert(3);
  assert(myset.count(3) != 1);
  for (i=0;i<10; i++)
  {
    cout << i;
    if (myset.count(i)>0)
      cout << " is an element of myset.\n";
    else 
      cout << " is not an element of myset.\n";
  }

  return 0;
}
