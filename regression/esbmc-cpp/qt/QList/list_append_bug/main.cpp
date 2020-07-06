// QList::back
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;

  myQList.append(10);
  int n = 10;
  while (myQList.back() != 0)
  {
    assert(myQList.back() != n--);
    myQList.push_back ( myQList.back() -1 );
  }

  return 0;
}
