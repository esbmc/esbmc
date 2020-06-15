// QLinkedList::push_front
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  QLinkedList<int> myQLinkedList;         // two ints with a value of 100
  myQLinkedList.push_back(100);
  myQLinkedList.push_back(100);
  assert(myQLinkedList.front() == 100);
  myQLinkedList.push_front (200);
  assert(myQLinkedList.front() == 200);
  myQLinkedList.push_front (300);
  assert(myQLinkedList.front() == 300);

  cout << "myQLinkedList contains:";
  for (QLinkedList<int>::iterator it=myQLinkedList.begin(); it!=myQLinkedList.end(); ++it)
    cout << " " << *it;

  cout << endl;
  return 0;
}
