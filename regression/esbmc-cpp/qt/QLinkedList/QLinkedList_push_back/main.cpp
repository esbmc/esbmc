// QLinkedList::push_back
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  QLinkedList<int> myQLinkedList;
  int myint;

  cout << "Please enter some integers (enter 0 to end):\n";

  do {
    cin >> myint;
    myQLinkedList.push_back (myint);
  } while (myint);

  cout << "myQLinkedList stores " << (int) myQLinkedList.size() << " numbers.\n";

  return 0;
}
