#include <QVector>
#include <cassert>

int main ()
{
    QVector<int> first;
    first << 1 << 2 << 3 << 4;
    first.insert(3, 2 ,99);
    assert(first.at(3) == 99);
    assert(first.at(4) == 99);
    assert(first.at(5) == 4);
    assert(first.size() == 6);
  return 0;
}

