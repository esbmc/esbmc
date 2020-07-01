#include <iostream>
#include <QLinkedList>
#include <QString>
#include <cassert>
using namespace std;

int main ()
{
    QLinkedList<QString> first;
    first << "sun" << "cloud" << "sun" << "rain";
    assert(first.last() == "sun");
  return 0;
}
