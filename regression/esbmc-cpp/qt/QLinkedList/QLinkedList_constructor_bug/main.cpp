// constructing QLinkedLists
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  // constructors used in the same order as described above:
  QLinkedList<int> first;                                // empty QLinkedList of ints
  assert(first.size() == 0);
  first.push_back(100);
  QLinkedList<int> fourth (first);                       // a copy of third
  assert(fourth.size() == 1);
  assert(fourth.back() == 100);

  QLinkedList<int>::iterator it = fourth.begin();
  
  assert(*it != 100);

  return 0;
}
