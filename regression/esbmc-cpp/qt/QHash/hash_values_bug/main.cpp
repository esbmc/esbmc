#include <cassert>
#include <QHash>
#include <QList>

int main ()
{
    QList<int> mylist;
    QHash<int, int> myQHash;
    QHash<int, int> :: const_iterator it;
    int iRet;

    myQHash[1] = 500;
    myQHash[2] = 300;
    myQHash[3] = 100;
    mylist = myQHash.values();
    assert(mylist.count() != 3);

    return 0;
}
