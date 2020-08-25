#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> list;
    list << "A" << "B" << "C" << "B" << "A";
    assert(list.contains("B"));
    assert(list.contains("A"));
    assert(list.contains("Z"));
  return 0;
}
