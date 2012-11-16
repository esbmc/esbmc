// map::insert
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::iterator it;
  pair<map<char,int>::iterator,bool> ret;

  // first insert function version (single parameter):
  mymap.insert ( pair<char,int>('a',100) );
  assert(mymap['a'] != 100);
  mymap.insert ( pair<char,int>('z',200) );
  assert(mymap['z'] != 200);
  ret=mymap.insert (pair<char,int>('z',500) ); 
  assert(ret.second != false);
  if (ret.second==false)
  {
    cout << "element 'z' already existed";
    cout << " with a value of " << ret.first->second << endl;
  }

  return 0;
}
