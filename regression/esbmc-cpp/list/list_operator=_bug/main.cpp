// assignment operator with lists
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> first (3);      // list of 3 zero-initialized ints
  list<int> second (5);     // list of 5 zero-initialized ints

  second=first;
  assert(second != first);
  assert(second.size() == 3);
  first=list<int>();

  cout << "Size of first: " << int (first.size()) << endl;
  cout << "Size of second: " << int (second.size()) << endl;
  return 0;
}
