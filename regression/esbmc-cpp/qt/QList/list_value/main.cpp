#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "A" << "B" << "C" << "D" << "E" << "F";
    assert(list.value(2) == "C");
    assert(list.size() == 6);
  return 0;
}
