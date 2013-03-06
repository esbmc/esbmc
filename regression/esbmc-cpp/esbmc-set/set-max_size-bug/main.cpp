#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int i;
  set<int> myset;
  
  assert(myset.max_size() < 1000);
  return 0;
}
