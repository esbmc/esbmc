#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "A" << "B" << "C" << "D" << "E" << "F";
    assert(list.takeAt(1) != "B");
    assert(list.size() == 5);
    // list: ["A", "E", "C", "D", "B", "F"]
  return 0;
}
