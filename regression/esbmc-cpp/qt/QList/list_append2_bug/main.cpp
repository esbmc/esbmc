// QList::back
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> first;
  QList<int> second;
  first.push_back(10);
  second.append(first);
    assert(second.size() != 1);
  int n = 10;
  while (second.back() != 0)
  {
    assert(second.back() == n--);
    second.push_back ( second.back() - 1 );
  }

  return 0;
}
