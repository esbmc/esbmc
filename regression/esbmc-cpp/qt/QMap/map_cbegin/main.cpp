#include <cassert>
#include <QMap>

int main ()
{
    QMap<int, int> myQMap;
    QMap<int, int> :: const_iterator it;

    myQMap[1] = 500;
    myQMap[2] = 300;
    myQMap[3] = 100;

    it = myQMap.cbegin();
    assert( myQMap.contains(it.key()) );

    return 0;
}
