#include <cassert>
#include <QMap>

int main ()
{
    QMap<int, int> myQMap1;

    myQMap1[1] = 500;
    myQMap1[2] = 300;
    myQMap1[3] = 100;

    assert(myQMap1.size() == 0);

    return 0;
}

