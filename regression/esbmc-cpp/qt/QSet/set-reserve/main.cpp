#include <iostream>
#include <cassert>
#include <QSet>
using namespace std;

int main ()
{
    QSet<int> set;
    set.reserve(2);
    for (int i = 0; i < 2; ++i)
        set.insert(i);
    assert(set.size() == 2);
  return 0;
}
