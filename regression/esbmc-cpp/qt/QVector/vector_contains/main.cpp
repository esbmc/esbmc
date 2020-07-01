#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "A" << "B" << "C" << "B" << "A";
    assert(list.contains("B"));
    assert(list.contains("A"));
    assert(list.contains("C"));
    //assert(list.indexOf("X") == -1);          // returns -1
  return 0;
}
