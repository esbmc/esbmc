// QMap::find
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
  QMap<char,int> myQMap;
  QMap<char,int>::iterator it;

  myQMap['a']=50;
  myQMap['b']=100;
  myQMap['c']=150;
  myQMap['d']=200;

  assert(myQMap.find('a').key() == 'a');
  assert(myQMap.find('a').value() == 50);
  
  assert(myQMap.find('b').key() == 'b');
  assert(myQMap.find('b').value() == 100);

  assert(myQMap.find('c').key() == 'c');
  assert(myQMap.find('c').value() == 150);
  
  assert(myQMap.find('d').key() == 'd');
  assert(myQMap.find('d').value() == 200);

  it=myQMap.find('b');
  myQMap.erase(it);

  // print content:
  cout << "elements in myQMap:" << endl;
  cout << "a => " << myQMap.find('a').value() << endl;
  cout << "c => " << myQMap.find('c').value() << endl;

  return 0;
}
