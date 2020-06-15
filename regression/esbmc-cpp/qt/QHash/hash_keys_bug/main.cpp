#include <cassert>
#include <QHash>
#include <QList>

int main ()
{
    QList<int> mylist;
    QHash<int, int> myQHash;
    QHash<int, int> :: const_iterator it;

    myQHash[1] = 500;
    myQHash[2] = 300;
    myQHash[3] = 100;
    mylist = myQHash.keys();
    assert(mylist.count() == 0);

    return 0;
}
