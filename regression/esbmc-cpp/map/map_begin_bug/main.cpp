// map::begin/end
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::iterator it;

  mymap['b'] = 100;
  mymap['a'] = 200;
  mymap['c'] = 300;
  it = mymap.begin();
 
  assert((*it).first != 'a');
  assert((*it).second == 200);
  it++;
  assert((*it).first == 'b');
  assert((*it).second != 100);
  it++;
  assert((*it).first == 'c');
  assert((*it).second != 300);
  
  // show content:
  for ( it=mymap.begin() ; it != mymap.end(); it++ )
    cout << (*it).first << " => " << (*it).second << endl;

  return 0;
}
