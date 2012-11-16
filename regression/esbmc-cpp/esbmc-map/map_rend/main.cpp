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
  rit = mymap.rend();
  rit++;
  assert(rit->first == 'z');
  assert(rit->second == 300);
  rit = mymap.rend();
  rit--;
  assert(rit->first == 'x');
  assert(rit->second == 100);
  
  // show content:
  for ( rit=mymap.rbegin() ; rit != mymap.rend(); rit++ )
    cout << rit->first << " => " << rit->second << endl;

  return 0;
}
