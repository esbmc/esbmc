#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    QString _default = "Z";
    list << "A" << "B" << "C" << "D" << "E" << "F";
    assert(list.at(1) == "B");
    assert(list.size() == 6);
  return 0;
}
