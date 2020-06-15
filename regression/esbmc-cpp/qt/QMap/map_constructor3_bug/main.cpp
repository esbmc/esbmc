// constructing QMaps
#include <iostream>
#include <QMap>
#include <map>
#include <cassert>
using namespace std;


int main ()
{
    map<char,int>first;
    
    first['a']=10;
    first['b']=30;
    first['c']=50;
    
    QMap<char,int>second(first);
    
    assert(second.size() != 3);
    return 0;
}
