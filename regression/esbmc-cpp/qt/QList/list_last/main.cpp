// QList::back
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;

  myQList.push_back(10);
  int n = 10;
  while (myQList.last() != 0)
  {
    assert(myQList.last() == n--);
    myQList.push_back ( myQList.last() -1 );
  }

  return 0;
}
