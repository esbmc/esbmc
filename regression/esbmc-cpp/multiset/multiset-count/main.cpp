#include <iostream>
#include <set>
#include <cassert>
using namespace std;

// clang++ main.cpp -o main 
// main: main.cpp:20: int main(): Assertion `myset.count(3) == 1' failed.
// Aborted (core dumped)

int main ()
{
  multiset<int> myset;
  int i;

  // set some initial values:
  for (i=1; i<5; i++) myset.insert(i*3);    // set: 3 6 9 12
  assert(myset.count(3) == 1);
  assert(myset.count(6) == 1);
  assert(myset.count(9) == 1);
  assert(myset.count(12) == 1);
  myset.insert(3);
  myset.insert(3);
  myset.insert(3);
  assert(myset.count(3) == 1); // Multisets allow duplicates by definition
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
