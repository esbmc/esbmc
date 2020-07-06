#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  QSet<int> myQSet;
  QSet<int>::iterator it;

  // QSet some initial values:
  for (int i=1; i<=5; i++) myQSet.insert(i*10);    // QSet: 10 20 30 40 50

  assert(myQSet.size() == 5);

  myQSet.insert(20);               // no new element inserted
  assert(myQSet.size() == 6);

  return 0;
}
