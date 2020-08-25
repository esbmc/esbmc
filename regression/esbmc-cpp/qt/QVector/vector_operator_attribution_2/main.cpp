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
    first << second;
    assert(first.size() == 5);
    assert(first.at(0) == "A");
    assert(first.at(1) == "B");
    assert(first.at(2) == "C");
    assert(first.at(3) == "D");
    assert(first.at(4) == "E");
  return 0;
}
