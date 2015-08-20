#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  set<int> myset;
  set<int>::iterator it;

  myset.insert (100);
  myset.clear();
  assert(myset.size() != 0);

  return 0;
}
