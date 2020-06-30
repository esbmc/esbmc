#include <iostream>
#include <QVector>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QVector<QString> first;
    first << "A" << "B" << "C";
    QVector<QString> second;
    second << "D" << "E";
    second += first;
    assert(second.size() == 5);
  return 0;
}
