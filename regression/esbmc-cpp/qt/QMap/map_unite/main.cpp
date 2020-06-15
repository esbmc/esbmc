// QMap::insert
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
    QMap<char,int> first;
    QMap<char,int> second;

  //key() insert function version (single parameter):
    first.insert ( 'a', 100 );
    first.insert ( 'b', 200 );
    first.insert ( 'c', 300 );

    second.insert ( 'd', 400 );
    second.insert ( 'e', 300 );
    second.insert ( 'f', 500 );
    
    first.unite(second);
    
    assert(first.size() == 6);
  return 0;
}
