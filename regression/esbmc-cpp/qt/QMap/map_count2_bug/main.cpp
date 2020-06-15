// QMap::size
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
  QMap<char,int> myQMap;
  myQMap['a']=101;
  myQMap['b']=202;
  myQMap['c']=302;
  assert(myQMap.count() != 3);
  cout << "myQMap.count() is " << (int) myQMap.count() << endl;

  return 0;
}
