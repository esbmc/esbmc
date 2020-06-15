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
    second += first;
    assert(second.size() != 5);
  return 0;
}
