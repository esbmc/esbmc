#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  QSet<int> myQSet;
  int i;

  // QSet some initial values:
  for (i=1; i<5; i++) myQSet.insert(i*3);    // QSet: 3 6 9 12
  assert(myQSet.count() == 4);
  myQSet.insert(3);
  myQSet.insert(3);
  myQSet.insert(3);
  cout << myQSet.count() << endl;
  assert(myQSet.count() == 7);
  return 0;
}
