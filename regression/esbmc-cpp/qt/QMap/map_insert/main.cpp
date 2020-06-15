// QMap::insert
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
  QMap<char,int> myQMap;
  QMap<char,int>::iterator it;

  //key() insert function version (single parameter):
  myQMap.insert ( 'a', 100 );
  assert(myQMap['a'] == 100);
  myQMap.insert ( 'z', 200 );
  assert(myQMap['z'] == 200);

  return 0;
}
