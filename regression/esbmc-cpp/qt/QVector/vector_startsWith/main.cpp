#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> list;
    list << "sun" << "cloud" << "sun" << "rain";
    assert(list.startsWith("sun"));
  return 0;
}
