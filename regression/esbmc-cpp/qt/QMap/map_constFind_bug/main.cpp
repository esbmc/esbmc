#include <cassert>
#include <QMap>

int main ()
{
    QMap<int, int> myQMap;
    QMap<int, int> :: const_iterator it;

    myQMap[1] = 500;
    myQMap[2] = 300;
    myQMap[3] = 100;

    it = myQMap.constFind(2);

    assert(it.key() != 2);
    assert(it.value() != 300);

    return 0;
}
