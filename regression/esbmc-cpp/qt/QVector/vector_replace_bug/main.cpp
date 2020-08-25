#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "sun" << "cloud" << "sun" << "rain";
    list.replace(0,"moon");
    assert(list.front() != "moon");
    // list: ["moon", "cloud", ,"sun", "rain"]
  return 0;
}
