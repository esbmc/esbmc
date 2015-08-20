// map::value_comp
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::iterator it;
  pair<char,int> highest;

  mymap['x']=1001;
  mymap['y']=2002;
  mymap['z']=3003;

  cout << "mymap contains:\n";

  highest=*mymap.rbegin();          // last element

  it=mymap.begin();
  do {
    cout << (*it).first << " => " << (*it).second << endl;
  } while ( mymap.value_comp()(*it++, highest) );
  assert(!( mymap.value_comp()(*it++, highest) ));
  return 0;
}
