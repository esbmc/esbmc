// constructing QMaps
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;


int main ()
{
    QMap<char,int>first;
    
    first['a']=10;
    first['b']=30;
    first['c']=50;
    first['d']=70;
    
    QMap<char,int>second(first);
    
    assert(second.size() == 4);
    return 0;
}
