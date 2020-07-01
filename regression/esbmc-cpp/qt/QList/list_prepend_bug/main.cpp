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
    list.prepend("three");
    assert(list.first() != "three");
    // list: ["three", "two", "one"]
  return 0;
}
