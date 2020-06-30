#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "A" << "B" << "C" << "B" << "A";
    assert(list.endsWith("A"));          // returns 1
  return 0;
}
