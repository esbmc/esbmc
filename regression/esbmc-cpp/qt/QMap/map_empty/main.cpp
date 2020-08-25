// QMap::empty
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
  QMap<char,int> myQMap;

  myQMap['a']=10;
  myQMap['b']=20;
  myQMap['c']=30;

  while (!myQMap.empty())
  {
     cout << myQMap.begin().key() << " => ";
     cout << myQMap.begin().value() << endl;
     myQMap.erase(myQMap.begin());
  }
  assert(myQMap.empty());
  return 0;
}
