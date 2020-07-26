#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list.prepend("one");
    list.prepend("two");
    list.removeFirst();
    assert(list.first() == "one");
  return 0;
}
