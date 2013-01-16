// map::clear
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::iterator it;

  mymap['x']=100;
  mymap['y']=200;
  mymap['z']=300;

  cout << "mymap contains:\n";
  for ( it=mymap.begin() ; it != mymap.end(); it++ )
    cout << (*it).first << " => " << (*it).second << endl;
  assert(mymap.size() == 3);
  mymap.clear();
  assert(mymap.size() == 0);
  mymap['a']=1101;
  mymap['b']=2202;
  assert(mymap.size() == 2);
  cout << "mymap contains:\n";
  for ( it=mymap.begin() ; it != mymap.end(); it++ )
    cout << (*it).first << " => " << (*it).second << endl;

  return 0;
}
