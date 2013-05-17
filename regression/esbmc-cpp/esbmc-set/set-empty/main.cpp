#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  set<int> myset;

  myset.insert(20);
  myset.insert(30);
  myset.insert(10);
  assert(myset.size() == 3);
  set<int>::iterator it = myset.begin();
  assert(*it == 10);
  cout << "myset contains:";
  while (!myset.empty())
  {
     cout << " " << *myset.begin();
     myset.erase(myset.begin());
  }
  assert(myset.begin() == myset.end());
  assert(myset.size() == 0);
  cout << endl;

  return 0;
}
