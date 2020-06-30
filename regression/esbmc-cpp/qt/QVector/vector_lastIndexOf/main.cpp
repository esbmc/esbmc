#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> first;
    first << "sun" << "cloud" << "sun" << "rain";
    assert(first.lastIndexOf("sun") == 2);
  return 0;
}
