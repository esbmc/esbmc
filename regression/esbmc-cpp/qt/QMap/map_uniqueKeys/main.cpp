// QMap::insert
#include <iostream>
#include <QMap>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QMap<char,int> myQMap;
  QMap<char,int>::iterator it;

  //key() insert function version (single parameter):
  myQMap.insert ( 'z', 100 );
  myQMap.insert ( 'a', 200 );

    QList<char> mylist = myQMap.uniqueKeys();
    
    assert(mylist.size() == 2);
    assert(mylist.front() == 'a');
    assert(mylist.back() == 'z');
  return 0;
}
