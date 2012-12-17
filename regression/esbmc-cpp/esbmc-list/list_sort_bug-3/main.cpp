#include <iostream>
#include <cassert>
#include <list>
using namespace std;

int main ()
{
  list<int> mylist;

  mylist.push_back (3);
  mylist.push_back (2);
  mylist.push_back (4);
  mylist.push_back (1);

  mylist.sort();

  assert(mylist.front() != 1);
  assert(mylist.back() == 4);
  assert(mylist.size() != 4);
  
  return 0;
}
