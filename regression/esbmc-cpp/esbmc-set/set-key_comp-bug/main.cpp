#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  set<int> myset;
  set<int>::key_compare mycomp;
  set<int>::iterator it,it2;
  int i,highest;

  mycomp = myset.key_comp();
  it = myset.begin();
  it2 = myset.begin();
  
  assert(mycomp(*it,*it2));


  return 0;
}
