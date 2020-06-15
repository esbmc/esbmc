// QMultiMap::find
#include <iostream>
#include <QMultiMap>
#include <cassert>
using namespace std;

int main ()
{
  QMultiMap<char,int> myQMultiMap;
  QMultiMap<char,int>::iterator it;

  myQMultiMap['a']=50;
  myQMultiMap['b']=100;
  myQMultiMap['c']=150;
  myQMultiMap['d']=200;

  assert(myQMultiMap.find('a',50).key() == 'a');
  assert(myQMultiMap.find('a',50).value() == 50);
  
  assert(myQMultiMap.find('b',100).key() == 'b');
  assert(myQMultiMap.find('b',100).value() == 100);

  assert(myQMultiMap.find('c',150).key() == 'c');
  assert(myQMultiMap.find('c',150).value() == 150);
  
  assert(myQMultiMap.find('d',200).key() == 'd');
  assert(myQMultiMap.find('d',200).value() == 200);

  it=myQMultiMap.find('b',100);
  myQMultiMap.erase(it);

  return 0;
}
