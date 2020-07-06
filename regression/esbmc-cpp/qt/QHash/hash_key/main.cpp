#include <cassert>
#include <QHash>


int main ()
{
    QHash<int, int> myQHash;
    QHash<int, int> :: const_iterator it;
    int iRet;

    myQHash[1] = 500;
    myQHash[2] = 300;
    myQHash[3] = 100;

    iRet = myQHash.key(100);

    assert(iRet == 3);

    return 0;
}
