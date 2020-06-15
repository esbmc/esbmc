#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "sun" << "cloud" << "sun" << "rain";
    list.removeOne("sun");
    assert(list.front() == "cloud");
    // list: ["cloud", ,"sun", "rain"]
  return 0;
}
