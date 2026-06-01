// reversing list
#include <iostream>
#include <cassert>
#include <list>
using namespace std;

int main ()
{
  list<int> mylist;
  list<int>::iterator it;

  for (int i=1; i<5; i++) mylist.push_back(i);

  mylist.reverse();

  // After reversing {1,2,3,4}, the head should be 4.  The original test
  // asserted `*it != 4`, which contradicts std::list::reverse semantics and
  // is why this case was marked KNOWNBUG.  Match the sibling list_reference-1
  // / list_reference-2 tests, which assert positive equality.
  it = mylist.begin();
  assert(*it == 4);

  cout << "*it: " << *it << endl;
  return 0;
}
