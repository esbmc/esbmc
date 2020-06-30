#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  QSet<int> myQSet;
  QSet<int>::iterator it;

  // insert some values:
  for (int i=1; i<10; i++) myQSet.insert(i*10);  // 10 20 30 40 50 60 70 80 90

  it=myQSet.begin();
  myQSet.erase (it);
  assert(myQSet.size() == 8);

  return 0;
}
