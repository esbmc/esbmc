#include <iostream>
#include <cassert>
#include <QSet>
#include <QString>
#include <QList>
using namespace std;

int main ()
{
    QList<QString> list;
    list << "Julia" << "Mike" << "Mike" << "Julia" << "Julia";
    
    QSet<QString> set = QSet<QString>::fromList(list);
    assert(set.contains("Felipe"));  // returns true
    
    return 0;
}
