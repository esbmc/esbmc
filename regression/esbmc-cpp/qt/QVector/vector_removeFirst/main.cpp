#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list.prepend("one");
    list.prepend("two");
    list.removeFirst();
    assert(list.first() == "one");
  return 0;
}
