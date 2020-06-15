#include <cassert>
#include <QMap>
#include <QList>

int main ()
{
    QList<int> mylist;
    QMap<int, int> myQMap;
    QMap<int, int> :: const_iterator it;
    int iRet;

    myQMap[1] = 500;
    myQMap[2] = 300;
    myQMap[3] = 100;
    mylist = myQMap.values();
    assert(mylist.count() != 3);

    return 0;
}
