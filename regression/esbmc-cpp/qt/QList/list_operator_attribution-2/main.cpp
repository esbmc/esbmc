#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> first;
    first << "A" << "B" << "C";
    QList<QString> second;
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
