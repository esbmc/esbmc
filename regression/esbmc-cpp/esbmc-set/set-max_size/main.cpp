#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int i;
  set<int> myset;
  if (myset.max_size()>1000)
  {
    for (i=0; i<1000; i++) myset.insert(i);
    cout << "The set contains 1000 elements.\n";
  }
  else cout << "The set could not hold 1000 elements.\n";
  assert(myset.size() == 1000);
  return 0;
}
