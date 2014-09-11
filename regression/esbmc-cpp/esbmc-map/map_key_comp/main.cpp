// map::key_comp
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::key_compare mycomp;
  map<char,int>::iterator it;
  char highest;

  mycomp = mymap.key_comp();

  mymap['a']=100;
  mymap['b']=200;
  mymap['c']=300;

  cout << "mymap contains:\n";

  highest=mymap.rbegin()->first;     // key value of last element

  it=mymap.begin();
  do {
    cout << (*it).first << " => " << (*it).second << endl;
  } while ( mycomp((*it++).first, highest) );
  assert(mycomp((*it++).first, highest));
  cout << endl;

  return 0;
}
