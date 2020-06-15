// QLinkedList::begin/end
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<int> myQLinkedList;
    myQLinkedList.push_back(75);
    myQLinkedList.push_back(23);
    myQLinkedList.push_back(65);
    myQLinkedList.push_back(42);
    myQLinkedList.push_back(13);
  QLinkedList<int>::iterator it;
  it = myQLinkedList.end();
  it--;
  assert(*it != 13);
  
  cout << endl;

  return 0;
}
