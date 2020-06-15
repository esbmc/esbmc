// QMap::count
#include <iostream>
#include <QMap>
#include <cassert>
using namespace std;

int main ()
{
  QMap<char,int> myQMap;
  char c;

  myQMap ['a']=101;
  myQMap ['c']=202;
  myQMap ['f']=303;
  char chararray1[3] = {'a', 'c', 'f'};
  char chararray2[4] = {'b', 'd', 'e', 'g'};
/*  for (c='a'; c<'h'; c++)
  {
    cout << c;
    if (myQMap.count(c)>0)
      cout << " is an element of myQMap.\n" << myQMap.count(c) << endl;
    else 
      cout << " is not an element of myQMap.\n" << myQMap.count(c) << endl;
      
  }*/
  int i;
  for (i = 0 ; i < 3 ; i++)
  	assert(!(myQMap.count(chararray1[i]) > 0));
  
  for (i = 0 ; i < 4 ; i++)
  	assert(myQMap.count(chararray2[i]) != 0);

  return 0;
}
