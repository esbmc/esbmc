#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "A" << "B" << "C" << "D" << "E" << "F";
    assert(list.takeLast() != "F");
    assert(list.size() != 5);
  return 0;
}
