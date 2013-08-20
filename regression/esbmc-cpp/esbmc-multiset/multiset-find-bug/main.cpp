#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  int b[5] = {10,20,30,40,50};
  multiset<int> myset(b,b+5);
  multiset<int>::iterator it;
  int i;
  // set some initial values:
//  for (i=1; i<=5; i++) myset.insert(i*10);    // set: 10 20 30 40 50
  assert(myset.size() == 5);
  it=myset.find(20);
  assert(*it == 20);
  myset.erase (it);
  myset.erase (myset.find(40));
  it = myset.begin();
  it++;it++;
  assert(*it != 50);

  cout << "myset contains:";
  for (it=myset.begin(); it!=myset.end(); it++)
    cout << " " << *it;
  cout << endl;

  return 0;
}
