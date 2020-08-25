#include <iostream>
#include <QLinkedList>
#include <QString>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> first;
    list<QString> second;
    first << "A" << "B" << "C" << "D" << "E" << "F";
    second = first.toStdList();
    assert(second.size() == 6);
  return 0;
}
