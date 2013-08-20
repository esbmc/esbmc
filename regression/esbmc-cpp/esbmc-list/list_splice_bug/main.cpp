// splicing lists
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist1, mylist2;
  list<int>::iterator it;

  // set some initial values:
  for (int i=1; i<=4; i++)
     mylist1.push_back(i);      // mylist1: 1 2 3 4

  for (int i=1; i<=3; i++)
     mylist2.push_back(i*10);   // mylist2: 10 20 30

  it = mylist1.begin();
  ++it;                         // points to 2

  mylist1.splice (it, mylist2); // mylist1: 1 10 20 30 2 3 4
                                // mylist2 (empty)
                                // "it" still points to 2 (the 5th element)
  assert(mylist1.size() == 7);
  assert(mylist2.empty());
  assert(*it == 2);
  it--;
  assert(*it != 30);
  it++;

  return 0;
}
