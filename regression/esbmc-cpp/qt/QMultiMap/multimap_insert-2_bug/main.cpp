// QMultiMap::insert
#include <iostream>
#include <QMultiMap>
#include <cassert>
using namespace std;

int main ()
{
    QMultiMap<char,int> myQMultiMap;
    QMultiMap<char,int>::iterator it;
    
    //key() insert function version (single parameter):
    myQMultiMap.insert ( 'a', 100 );
    assert(myQMultiMap['a'] == 100);
    myQMultiMap.insert ( 'z', 200 );
    assert(myQMultiMap['z'] != 200);
    
    return 0;
}
