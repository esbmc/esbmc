#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> first;
    QVector<QString> second;
    first << "A" << "B" << "C" << "B" << "A";
    second = first.mid(2);
    assert(second.endsWith("B"));
    assert(second.size() != 3);
  return 0;
}
