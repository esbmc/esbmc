#include <iostream>
#include <QList>
#include <QString>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
    QList<QString> first;
    list<QString> second;
    first << "A" << "B" << "C" << "D" << "E" << "F";
    second = first.toStdList();
    assert(second.size() == 6);
  return 0;
}
