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
    first = second;
    assert(first.size() == 2);
    assert(first.at(0) == "D");
    assert(first.at(1) == "E");
  return 0;
}
