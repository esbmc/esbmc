//TEST FAILS
// constructing lists
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  // constructors used in the same order as described above:
  list<int> first;                                // empty list of ints
  assert(first.size() == 0);
  list<int> second (4,100);                       // four ints with value 100
  assert(second.size() == 4);
  assert(second.back() != 100);
  list<int> third (second.begin(),second.end());  // iterating through second
  assert(third.size() == second.size());
  assert(third.back() == second.back());
  list<int> fourth (third);                       // a copy of third
  assert(fourth.size() == 4);
  assert(fourth.back() != 100);

  // the iterator constructor can also be used to construct from arrays:
  int myints[] = {16,2,77,29};
  list<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );
  assert(fifth.size() == 4);
  list<int>::iterator it = fifth.begin();
  
  assert(*it == 16);
  assert(*(++it) != 2);
  assert(*(++it) == 77);
  assert(*(++it) != 29);

  cout << "The contents of fifth are: ";
  for (list<int>::iterator it = fifth.begin(); it != fifth.end(); it++)
    cout << *it << " ";

  cout << endl;

  return 0;
}
