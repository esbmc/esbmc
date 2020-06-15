#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "D" << "E";
    list += "F";
    assert(list.size() != 3);
  return 0;
}
