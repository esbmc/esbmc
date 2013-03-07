#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  set<int> myset;
  set<int>::iterator it;
  int i;
  assert(myset.size() == 0);
  // set some initial values:
  for (i=1; i<=5; i++) myset.insert(i*10);    // set: 10 20 30 40 50
  assert(myset.size() == 5);
  i = 10;
  for (it = myset.begin(); it != myset.end(); it++)
  {
    assert(*it == i);
    i+=10;
   }
  it=myset.find(20);
  assert(*it == 20);
  myset.erase (it);
  myset.erase (myset.find(40));
  it = myset.begin();
  it++;it++;
  assert(*it == 50);

  cout << "myset contains:";
  for (it=myset.begin(); it!=myset.end(); it++)
    cout << " " << *it;
  cout << endl;

  return 0;
}
