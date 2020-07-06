// QLinkedList::empty
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  QLinkedList<int> myQLinkedList;
  int sum (0);

  for (int i=1;i<=10;i++) myQLinkedList.push_back(i);

  assert(!myQLinkedList.empty());
  while (!myQLinkedList.empty())
  {
     sum += myQLinkedList.front();
     myQLinkedList.pop_front();
  }
  assert(!myQLinkedList.empty()||(myQLinkedList.size() != 0));
  cout << "total: " << sum << endl;
  
  return 0;
}
