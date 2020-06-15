#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> first;
    QList<QString> second;
    first << "A" << "B" << "C" << "B" << "A";
    second = first.mid(2);
    assert(second.endsWith("A"));
    assert(second.size() == 3);
  return 0;
}
