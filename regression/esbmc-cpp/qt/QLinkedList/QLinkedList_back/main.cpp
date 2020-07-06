//QLinkedList::back
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  QLinkedList<int> myQLinkedList;

  myQLinkedList.push_back(10);
  int n = 10;
  while (myQLinkedList.back() != 0)
  {
    assert(myQLinkedList.back() == n--);
    myQLinkedList.push_back ( myQLinkedList.back() -1 );
  }

  cout << "myQLinkedList contains:";
  for (QLinkedList<int>::iterator it=myQLinkedList.begin(); it!=myQLinkedList.end() ; ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
