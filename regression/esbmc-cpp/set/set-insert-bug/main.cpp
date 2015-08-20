#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  set<int> myset;
  set<int>::iterator it;
  pair<set<int>::iterator,bool> ret;

  // set some initial values:
  for (int i=1; i<=5; i++) myset.insert(i*10);    // set: 10 20 30 40 50

  assert(myset.size() == 5);

  ret = myset.insert(20);               // no new element inserted
  assert(!(ret.second));
  assert(*ret.first == 20);
  if (ret.second==false) it=ret.first;  // "it" now points to element 20

  myset.insert (it,25);                 // max efficiency inserting
  myset.insert (it,24);                 // max efficiency inserting
  myset.insert (it,26);                 // no max efficiency inserting

  int myints[]= {5,10,15};              // 10 already in set, not inserted
  myset.insert (myints,myints+3);
  assert(myset.size() == 10);
  cout << "myset contains:";
 
  it = myset.begin();
  assert(*it == 5);
  it++;
  assert(*it == 10);
  it++;
  assert(*it == 15);
  it++;
  assert(*it != 20);
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
  assert(*it != 50);
  return 0;
}
