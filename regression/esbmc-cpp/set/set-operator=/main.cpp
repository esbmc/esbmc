#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int myints[]={ 12,82,37,64,15 };
  set<int> first (myints,myints+5);   // set with 5 ints
  set<int> second;                    // empty set
  assert(second.size() == 0);
  second=first;                       // now second contains the 5 ints
  assert(second.size() == 5);
  set<int>::iterator it = second.begin();
  assert(*it == 12);
  ++it;
  assert(*it == 15);
  ++it;
  assert(*it == 37);
  ++it;
  assert(*it == 64);
  ++it;
  assert(*it == 82);
  first=set<int>();                   // and first is empty
  assert(first.size() == 0);
  cout << "Size of first: " << int (first.size()) << endl;
  cout << "Size of second: " << int (second.size()) << endl;
  return 0;
}
