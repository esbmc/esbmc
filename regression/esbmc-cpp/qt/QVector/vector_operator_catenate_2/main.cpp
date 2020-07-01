#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "D" << "E";
    list += "F";
    assert(list.size() == 3);
  return 0;
}
