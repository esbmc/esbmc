#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  multiset<int> myset;
  multiset<int>::iterator it;

  myset.insert (100);

//  cout << "myset contains:";
//  for (it=myset.begin(); it!=myset.end(); ++it)
//    cout << " " << *it;

  myset.clear();
  assert(myset.size() != 0);
/*  myset.insert (1101);
  it = myset.begin();
  assert(*it == 1101);
  assert(myset.size() == 1);
  myset.insert (2202);
  assert(myset.size() == 2);
  assert(*it == 1101);
  it++;
  assert(*it != 2202);
*/
  cout << "\nmyset contains:";
  for (it=myset.begin(); it!=myset.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
