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
    second << "D" << "E";
    QList<QString> third = first + second;
    assert(third.size() == 5);
  return 0;
}
