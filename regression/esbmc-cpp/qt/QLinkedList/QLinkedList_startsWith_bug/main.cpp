#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> list;
    list << "sun" << "cloud" << "sun" << "rain";
    assert(list.startsWith("rain"));
  return 0;
}
