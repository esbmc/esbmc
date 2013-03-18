#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int myints[] = {75,23,65,42,13};
  int mySecondInts[] = {13,23,42,65,75};
  int i;
  set<int> myset (myints,myints+5);
  assert(myset.size() == 5);
  set<int>::iterator it;

  cout << "myset contains:" << endl;/*
  for ( it=myset.begin(), i=0 ; it != myset.end(); it++, i++ ){
    cout << " " << *it;
    assert(*it == mySecondInts[i]);
  }*/
  for (i = 0;i < 5;i++) myset.insert(i);
  assert(0);
  it = myset.find(2);
  assert(*it != 2);
  cout << endl;

  return 0;
}
