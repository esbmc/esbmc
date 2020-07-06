#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> list;
    list << "A" << "B" << "C" << "D" << "E" << "F";
    list.swap(1, 4);
    assert(list.at(4) == "B");
    // list: ["A", "E", "C", "D", "B", "F"]
  return 0;
}
