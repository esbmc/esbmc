#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "sun" << "cloud" << "sun" << "rain";
    assert(list.startsWith("sun"));
  return 0;
}
