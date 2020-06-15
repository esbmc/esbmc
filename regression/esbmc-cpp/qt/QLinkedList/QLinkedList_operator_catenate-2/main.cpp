#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> list;
    list << "D" << "E";
    list += "F";
    assert(list.size() == 3);
  return 0;
}
