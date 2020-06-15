#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "A" << "B" << "C" << "D" << "E" << "F";
    assert(list.takeAt(1) == "B");
    assert(list.size() == 5);
    // list: ["A", "E", "C", "D", "B", "F"]
  return 0;
}
