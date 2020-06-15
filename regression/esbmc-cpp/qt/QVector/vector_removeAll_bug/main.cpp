#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "sun" << "cloud" << "sun" << "rain";
    list.removeAll("sun");
    assert(list.size() != 2);
    // list: ["cloud", "rain"]
  return 0;
}
