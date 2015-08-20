// map::rbegin/rend
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::reverse_iterator rit;

  mymap['x'] = 100;
  mymap['y'] = 200;
  mymap['z'] = 300;
  rit = mymap.rbegin();
  
  assert(rit->first != 'z');
  assert(rit->second != 300);
  
  cout << rit->first << " => " << rit->second << endl;

  // show content:
  for ( rit=mymap.rbegin() ; rit != mymap.rend(); rit++ )
    cout << rit->first << " => " << rit->second << endl;

  return 0;
}
