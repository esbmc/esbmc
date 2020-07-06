// QLinkedList::front
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  QLinkedList<int> myQLinkedList;

  myQLinkedList.push_back(77);
  myQLinkedList.push_back(22);
  assert(myQLinkedList.first() != 77);
  // now front equals 77, and back 22

  myQLinkedList.first() -= myQLinkedList.back();
  assert(myQLinkedList.first() == 55);
  cout << "myQLinkedList.front() is now " << myQLinkedList.first() << endl;

  return 0;
}
