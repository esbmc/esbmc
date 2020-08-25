#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> first;
    first << "sun" << "cloud" << "sun" << "rain";
    first.insert(2, "moon");
    assert(first.at(2) == "moon");
    assert(first.size() == 5);
  return 0;
}
