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
  assert(myQLinkedList.front() != 77);
  // now front equals 77, and back 22

  myQLinkedList.front() -= myQLinkedList.back();
  assert(myQLinkedList.front() == 55);
  cout << "myQLinkedList.front() is now " << myQLinkedList.front() << endl;

  return 0;
}
