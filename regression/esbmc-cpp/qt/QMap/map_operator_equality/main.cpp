#include <cassert>
#include <QMap>

int main ()
{
    QMap<int, int> myQMap1;
    QMap<int, int> myQMap2;
    QMap<int, int> :: const_iterator it;

    myQMap1[1] = 500;
    myQMap1[2] = 300;
    myQMap1[3] = 100;

    myQMap2[1] = 500;
    myQMap2[2] = 300;
    myQMap2[3] = 100;

    assert(myQMap1 == myQMap2);


    return 0;
}
