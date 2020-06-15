// QMap::begin/end
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
  QMap<char,int> myQMap;
  QMap<char,int>::iterator it;

  myQMap['b'] = 100;
  myQMap['a'] = 200;
  myQMap['c'] = 300;
 it = myQMap.end();
 it--;
    assert(it.key() != 'c');
    assert(it.value() != 300);

  // show content:
  for ( it=myQMap.begin() ; it != myQMap.end(); it++ )
    cout << it.key() << " => " << it.value() << endl;

  return 0;
}
