// QLinkedList::begin
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  int myints[] = {75,23,65,42,13};
  QLinkedList<int> myQLinkedList;
  for(int i = 0; i < 5; i++)
    myQLinkedList.push_back(myints[i]);
  QLinkedList<int>::const_iterator it;

  it = myQLinkedList.cbegin();
  assert(*it != 75);
  
  cout << endl;

  return 0;
}
