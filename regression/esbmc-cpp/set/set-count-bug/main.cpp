#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  int b[3] = {3,6,9};
  set<int> myset(b,b+3);

  // set some initial values:


  assert(myset.count(3) != 1);
  return 0;
}
