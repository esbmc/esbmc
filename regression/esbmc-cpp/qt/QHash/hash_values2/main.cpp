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
    myQHash[3] = 300;
    myQHash[4] = 100;
    mylist = myQHash.values(300);
    assert(mylist.count() == 2);

    return 0;
}
