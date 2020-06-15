// clearing QLinkedLists
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  QLinkedList<int> myQLinkedList;
  QLinkedList<int>::iterator it;

  myQLinkedList.push_back (100);
  myQLinkedList.push_back (200);
  myQLinkedList.push_back (300);

  cout << "myQLinkedList contains:";
  for (it=myQLinkedList.begin(); it!=myQLinkedList.end(); ++it)
    cout << " " << *it;
  assert(myQLinkedList.size() == 3);
  myQLinkedList.clear();
  assert(myQLinkedList.size() == 0);
  myQLinkedList.push_back (1101);
  myQLinkedList.push_back (2202);
  assert(myQLinkedList.size() == 2);
  cout << "\nmyQLinkedList contains:";
  for (it=myQLinkedList.begin(); it!=myQLinkedList.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
