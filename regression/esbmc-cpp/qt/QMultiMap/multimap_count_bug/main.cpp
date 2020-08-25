// QMultiMap::count
#include <iostream>
#include <QMultiMap>
#include <cassert>
using namespace std;

int main ()
{
  QMultiMap<char,int> myQMultiMap;
  char c;

  myQMultiMap ['a']=101;
  myQMultiMap ['c']=202;
  myQMultiMap ['f']=303;
  char chararray1[3] = {'a', 'c', 'f'};
  char chararray2[4] = {'b', 'd', 'e', 'g'};
/*  for (c='a'; c<'h'; c++)
  {
    cout << c;
    if (myQMultiMap.count(c)>0)
      cout << " is an element of myQMultiMap.\n" << myQMultiMap.count(c) << endl;
    else 
      cout << " is not an element of myQMultiMap.\n" << myQMultiMap.count(c) << endl;
      
  }*/
  int i;
  for (i = 0 ; i < 3 ; i++)
  	assert(!(myQMultiMap.count(chararray1[i]) > 0));
  
  for (i = 0 ; i < 4 ; i++)
  	assert(myQMultiMap.count(chararray2[i]) != 0);

  return 0;
}
