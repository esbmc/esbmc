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
    list.removeAt(0);
    assert(list.first() == "one");
    // list: ["three", "two", "one"]
  return 0;
}
