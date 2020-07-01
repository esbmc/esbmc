#include <cassert>
#include <QMultiMap>

int main ()
{
    QMultiMap<int, int> myQMultiMap;
    QMultiMap<int, int> :: const_iterator it;

    myQMultiMap[1] = 500;
    myQMultiMap[2] = 300;
    myQMultiMap[3] = 100;

    it = myQMultiMap.cbegin();

    assert(myQMultiMap.contains(it.key()));

    return 0;
}
