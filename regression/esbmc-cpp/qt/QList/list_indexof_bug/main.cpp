#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "A" << "B" << "C" << "B" << "A";
    assert(list.indexOf("B") == 1);          // returns 1
    assert(list.indexOf("B", 1) == 1);       // returns 1
    assert(list.indexOf("B", 2) == 3);       // returns 3
    assert(list.indexOf("X") != -1);         // returns -1
  return 0;
}
