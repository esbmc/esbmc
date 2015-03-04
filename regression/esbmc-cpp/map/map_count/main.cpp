// map::count
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  char c;

  mymap ['a']=101;
  mymap ['c']=202;
  mymap ['f']=303;
  char chararray1[3] = {'a', 'c', 'f'};
  char chararray2[4] = {'b', 'd', 'e', 'g'};
/*  for (c='a'; c<'h'; c++)
  {
    cout << c;
    if (mymap.count(c)>0)
      cout << " is an element of mymap.\n" << mymap.count(c) << endl;
    else 
      cout << " is not an element of mymap.\n" << mymap.count(c) << endl;
      
  }*/
  int i;
  for (i = 0 ; i < 3 ; i++)
  	assert(mymap.count(chararray1[i]) > 0);
  
  for (i = 0 ; i < 4 ; i++)
  	assert(mymap.count(chararray2[i]) == 0);

  return 0;
}
