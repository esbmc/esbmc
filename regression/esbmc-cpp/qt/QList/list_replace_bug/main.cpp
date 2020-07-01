#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "sun" << "cloud" << "sun" << "rain";
    list.replace(0,"moon");
    assert(list.front() != "moon");
    // list: ["moon", "cloud", ,"sun", "rain"]
  return 0;
}
