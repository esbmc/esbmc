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

    it = myQHash.cbegin();
    it++;
    it++;

    iRet = myQHash.count(it.key());

    assert(iRet == 0);

    return 0;
}
