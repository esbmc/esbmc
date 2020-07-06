#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> first;
    first << "A" << "B" << "C";
    QLinkedList<QString> second;
    second << "D" << "E";
    assert(!(first != second));
  return 0;
}
