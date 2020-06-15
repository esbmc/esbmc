#include <cassert>
#include <QMap>
#include <QList>

int main ()
{
    QList<int> mylist;
    QMap<int, int> myQMap;
    QMap<int, int> :: const_iterator it;

    myQMap[1] = 500;
    myQMap[2] = 300;
    myQMap[3] = 100;
    mylist = myQMap.keys();
    assert(mylist.count() == 3);

    return 0;
}
