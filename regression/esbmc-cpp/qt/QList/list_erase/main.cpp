// erasing from QList
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  unsigned int i;
  QList<unsigned int> myQList;
  QList<unsigned int>::iterator it1,it2;

  // set some values:
  for (i=1; i<10; i++) myQList.push_back(i*10);

                               // 10 20 30 40 50 60 70 80 90
    it1 = it2 = myQList.begin(); // ^^
    it2++;
    it2++;
    it2++;
    it2++;
    it2++;
    it2++;
  assert(*it2 == 70);
  ++it1;                       //    ^              ^

  it1 = myQList.erase (it1);   // 10 30 40 50 60 70 80 90
  assert(*it1 == 30);
  assert(myQList.size() == 8);
                               //    ^           ^

  it2 = myQList.erase (it2);   // 10 30 40 50 60 80 90
                               //    ^           ^
  assert(*it2 == 80);
  assert(myQList.size() == 7);

  ++it1;                      //       ^        ^
  --it2;                      //       ^     ^

  myQList.erase (it1,it2);     // 10 30 60 80 90
                              //        ^
  assert(myQList.size() == 5);

  assert(*it2 == 60);

  return 0;
}
