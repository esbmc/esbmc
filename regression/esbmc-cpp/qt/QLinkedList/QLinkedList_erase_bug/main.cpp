// erasing from QLinkedList
#include <iostream>
#include <QLinkedList>
#include <cassert>
using namespace std;

int main ()
{
  unsigned int i;
  QLinkedList<unsigned int> myQLinkedList;
  QLinkedList<unsigned int>::iterator it1,it2;

  // set some values:
  for (i=1; i<10; i++) myQLinkedList.push_back(i*10);

                              // 10 20 30 40 50 60 70 80 90
  it1 = it2 = myQLinkedList.begin(); // ^^
  advance (it2,6);            // ^                 ^
  assert(*it2 != 70);
  ++it1;                      //    ^              ^

  it1 = myQLinkedList.erase (it1);   // 10 30 40 50 60 70 80 90
  assert(*it1 != 30);
  assert(myQLinkedList.size() == 8);
                              //    ^           ^

  it2 = myQLinkedList.erase (it2);   // 10 30 40 50 60 80 90
                              //    ^           ^
  assert(*it2 != 80);
  assert(myQLinkedList.size() == 7);

  ++it1;                      //       ^        ^
  --it2;                      //       ^     ^

  myQLinkedList.erase (it1,it2);     // 10 30 60 80 90
                              //        ^
  assert(myQLinkedList.size() == 5);

  assert(*it2 != 60);

  cout << "myQLinkedList contains:";
  for (it1=myQLinkedList.begin(); it1!=myQLinkedList.end(); ++it1)
    cout << " " << *it1;
  cout << endl;

  return 0;
}
