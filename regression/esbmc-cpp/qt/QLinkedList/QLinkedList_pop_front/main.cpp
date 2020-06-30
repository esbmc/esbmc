// QLinkedList::pop_front
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  QLinkedList<int> myQLinkedList;
  myQLinkedList.push_back (100);
  myQLinkedList.push_back (200);
  myQLinkedList.push_back (300);
  assert(myQLinkedList.front() == 100);
  
  int n = 100;
  
  cout << "Popping out the elements in myQLinkedList:";
  while (!myQLinkedList.empty())
  {
    assert(myQLinkedList.front() == n);
    cout << " " << myQLinkedList.front();
    myQLinkedList.pop_front();
    n +=100;
  }
  assert(myQLinkedList.empty());
  cout << "\nFinal size of myQLinkedList is " << int(myQLinkedList.size()) << endl;

  return 0;
}
