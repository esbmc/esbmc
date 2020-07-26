#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> first;
    first.append("A");
    assert(first.isEmpty());
  return 0;
}
