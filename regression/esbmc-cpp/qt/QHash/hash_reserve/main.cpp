#include <cassert>
#include <QHash>

int main ()
{
    QHash<int, int> myQHash;
    QHash<int, int> :: const_iterator it;
    int i;

    myQHash.reserve(3);

    for (i = 0; i < 3; i++) {
        myQHash.insert(i,i+100);
    }

    assert(myQHash.size() == 3);

    return 0;
}
