#include <cassert>
#include <QHash>

int main ()
{
    QHash<int, int> myQHash;
    QHash<int, int> :: const_iterator it;

    myQHash[1] = 500;
    myQHash[2] = 300;
    myQHash[3] = 100;

    it = myQHash.constFind(2);

    assert(it.key() == 2);
    assert(it.value() == 300);

    return 0;
}

