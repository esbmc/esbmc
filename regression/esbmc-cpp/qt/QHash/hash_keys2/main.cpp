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
    myQHash[4] = 100;
    myQHash[5] = 100;
    myQHash[6] = 100;
    
    mylist = myQHash.keys(100);
    assert(mylist.count() == 4);

    return 0;
}
