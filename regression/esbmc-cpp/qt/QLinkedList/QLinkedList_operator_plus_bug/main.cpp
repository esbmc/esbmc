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
    QLinkedList<QString> third = first + second;
    assert(third.size() != 5);
  return 0;
}
