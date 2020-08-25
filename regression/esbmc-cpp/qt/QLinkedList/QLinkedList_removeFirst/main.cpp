#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> list;
    list.prepend("one");
    list.prepend("two");
    list.removeFirst();
    assert(list.first() == "one");
  return 0;
}
