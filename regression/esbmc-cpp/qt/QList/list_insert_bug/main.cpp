#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> first;
    first << "sun" << "cloud" << "sun" << "rain";
    first.insert(2, "moon");
    assert(first.at(2) != "moon");
    assert(first.size() != 5);
  return 0;
}
