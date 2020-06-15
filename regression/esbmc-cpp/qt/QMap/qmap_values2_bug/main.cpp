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
    myQMap[3] = 300;
    myQMap[4] = 100;
    mylist = myQMap.values(300);
    assert(mylist.count() != 2);

    return 0;
}
