#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> first;
    first << "A" << "B" << "C";
    QVector<QString> second;
    second << "D" << "E";
    first = second;
    assert(first.size() != 2);
    assert(first.at(0) != "D");
    assert(first.at(1) != "E");
  return 0;
}
