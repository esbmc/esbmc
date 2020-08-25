#include <cassert>
#include <QMap>

int main ()
{
    QMap<int, int> myQMap;
    QMap<int, int> :: const_iterator it;

    myQMap[1] = 500;
    myQMap[3] = 300;
    myQMap[5] = 100;

    it = myQMap.lowerBound(4);
    assert( myQMap.contains(it.key()) );

    return 0;
}
