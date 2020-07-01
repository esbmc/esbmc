#include <iostream>
#include <QList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> first;
    first << "A" << "B" << "C";
    QList<QString> second;
    second << "A" << "B" << "C";
    assert(!(first == second));
  return 0;
}
