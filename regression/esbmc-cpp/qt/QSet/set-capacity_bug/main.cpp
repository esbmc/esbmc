#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  QSet<int> myQSet;

  myQSet.insert (100);
  myQSet.insert (200);
  myQSet.insert (300);

  assert(myQSet.capacity() < 0);

  return 0;
}
