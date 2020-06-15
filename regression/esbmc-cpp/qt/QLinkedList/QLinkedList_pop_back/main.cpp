// QLinkedList::pop_back
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  QLinkedList<int> myQLinkedList;
  int sum (0);
  myQLinkedList.push_back (100);
  myQLinkedList.push_back (200);
  myQLinkedList.push_back (300);
  assert(myQLinkedList.back() == 300);
  int n = 3;
  while (!myQLinkedList.empty())
  {
    assert(myQLinkedList.back() == n*100);
    sum+=myQLinkedList.back();
    myQLinkedList.pop_back();
    n--;
  }

  cout << "The elements of myQLinkedList summed " << sum << endl;

  return 0;
}
