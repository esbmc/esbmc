// QList::back
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;
    assert(myQList.isEmpty());
    myQList.push_back(10);
    myQList.push_back(10);
    assert(myQList.isEmpty());
  return 0;
}
