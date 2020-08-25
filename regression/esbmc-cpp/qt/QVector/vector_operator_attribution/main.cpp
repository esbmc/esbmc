#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "D" << "E";
    assert(list.size() == 2);
    assert(list.at(0) == "D");
    assert(list.at(1) == "E");
  return 0;
}
