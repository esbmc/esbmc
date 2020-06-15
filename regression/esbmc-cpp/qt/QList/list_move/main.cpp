#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "A" << "B" << "C" << "D" << "E" << "F";
    list.move(1, 4);
    assert(list.at(4) == "B");
    // list: ["A", "C", "D", "E", "B", "F"]
  return 0;
}
