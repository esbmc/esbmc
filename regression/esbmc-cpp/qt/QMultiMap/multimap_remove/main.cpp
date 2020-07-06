#include <cassert>
#include <QMultiMap>

int main ()
{
    QMultiMap<int, int> myQMultiMap;
    QMultiMap<int, int> :: const_iterator it;

    myQMultiMap[1] = 500;
    myQMultiMap[2] = 300;
    myQMultiMap[3] = 100;

    myQMultiMap.remove(2,300);
    
    it = myQMultiMap.find(2,300);

    assert(it.key() != 2);
    assert(it.value() != 300);

    return 0;
}
