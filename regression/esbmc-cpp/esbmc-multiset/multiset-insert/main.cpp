#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  multiset<int> myset;
  multiset<int>::iterator it;

  // set some initial values:
  for (int i=1; i<=5; i++) myset.insert(i*10);    // set: 10 20 30 40 50

  assert(myset.size() == 5);

  it = myset.insert(20);               // no new element inserted

  myset.insert (it,25);                 // max efficiency inserting
  myset.insert (it,24);                 // max efficiency inserting
  myset.insert (it,26);                 // no max efficiency inserting

  int myints[]= {5,10,15};              // 10 already in set, not inserted


  it = myset.begin();
  assert(*it == 10);
  it++;
  assert(*it == 20);
  it++;
  assert(*it == 20);
  it++;
  assert(*it == 24);
  it++;
  assert(*it == 25);
  it++;
  assert(*it == 26);
  it++;
  assert(*it == 30);
  it++;
  assert(*it == 40);
  it++;
  assert(*it == 50);
  return 0;
}
