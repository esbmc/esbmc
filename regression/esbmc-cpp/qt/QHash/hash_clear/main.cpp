#include <cassert>
#include <QHash>

int main ()
{
    QHash<int, int> myQHash;
    QHash<int, int> :: iterator it;

    for(int i = 0; i < 5; i++)
    {
        myQHash[i] = i;
    }

    myQHash.clear();

    assert(myQHash.capacity() == 0);

    return 0;
}
