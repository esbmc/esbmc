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
    myQMap[4] = 100;
    myQMap[5] = 100;
    myQMap[6] = 100;
    
    mylist = myQMap.keys(100);
    assert(mylist.count() == 6);

    return 0;
}
