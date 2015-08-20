#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  multiset<int> myset;
  multiset<int>::iterator it;

  // set some initial values:

  int myints[]= {5,10,15};              // 10 already in set, not inserted
  myset.insert (myints,myints+3);
  assert(myset.size() == 3);
  
  it = myset.begin();
  assert(*it == 5);
  it++;
  assert(*it == 10);
  it++;
  assert(*it == 15);
  
  return 0;
}
