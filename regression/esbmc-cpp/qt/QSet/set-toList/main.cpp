#include <cassert>
#include <QSet>
#include <QString>
#include <QList>
using namespace std;

int main ()
{
    QSet<QString> set;
    set << "red" << "green" << "blue" << "black";
    
    QList<QString> list = set.toList();
    assert(list.size() == 4);  // returns true

  return 0;
}
