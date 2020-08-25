// QMap::clear
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
  QMap<char,int> myQMap;
  QMap<char,int>::iterator it;

  myQMap['x']=100;
  myQMap['y']=200;
  myQMap['z']=300;

  cout << "myQMap contains:\n";
  for ( it=myQMap.begin() ; it != myQMap.end(); it++ )
    cout << it.key() << " => " << it.value() << endl;
  assert(myQMap.size() == 3);
  myQMap.clear();
  assert(myQMap.size() != 0);
  myQMap['a']=1101;
  myQMap['b']=2202;
  assert(myQMap.size() == 2);
  cout << "myQMap contains:\n";
  for ( it=myQMap.begin() ; it != myQMap.end(); it++ )
    cout << it.key() << " => " << it.value() << endl;

  return 0;
}
