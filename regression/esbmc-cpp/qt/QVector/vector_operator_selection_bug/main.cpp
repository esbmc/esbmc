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
    assert(list[1] == "C");
    assert(list.size() != 6);
  return 0;
}
