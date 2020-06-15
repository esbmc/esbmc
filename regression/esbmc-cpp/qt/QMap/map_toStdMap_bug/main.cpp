#include <cassert>
#include <QMap>
#include <iostream>

using namespace std;

int main ()
{
    QMap<int, int> myQMap1;
    map<int, int> myMap2;
    QMap<int, int> :: const_iterator it;

    myQMap1[1] = 500;
    myQMap1[2] = 300;
    myQMap1[3] = 100;

    myMap2 = myQMap1.toStdMap();
    assert( myMap2.empty() );

    return 0;
}
