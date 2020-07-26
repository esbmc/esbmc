#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> first;
    first << "sun" << "cloud" << "sun" << "rain";
    assert(first.lastIndexOf("sun") == 3);
  return 0;
}
