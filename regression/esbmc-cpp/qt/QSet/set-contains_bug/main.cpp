#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  QSet<int> myQSet;
  QSet<int>::iterator it;

  myQSet.insert (100);
  myQSet.insert (200);
  myQSet.insert (300);

  assert(myQSet.contains(400));

  return 0;
}
