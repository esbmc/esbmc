#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    QString _default = "Z";
    list << "A" << "B" << "C" << "D" << "E" << "F";
    assert(list.value(10, _default) != "Z");
    assert(list.size() != 6);
  return 0;
}
