#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> list;
    list << "A" << "B" << "C" << "B" << "A";
    assert(list.endsWith("A"));          // returns 1
  return 0;
}
