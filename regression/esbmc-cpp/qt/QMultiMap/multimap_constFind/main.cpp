#include <cassert>
#include <QMultiMap>

int main ()
{
    QMultiMap<int, int> myQMultiMap;
    QMultiMap<int, int> :: const_iterator it;

    myQMultiMap.insert(1,500);
    myQMultiMap.insert(2,300);
    myQMultiMap.insert(3,100);

    it = myQMultiMap.find(2, 300);

    assert(it.key() == 2);
    assert(it.value() == 300);

    return 0;
}

