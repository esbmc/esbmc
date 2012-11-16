// accessing mapped values
#include <iostream>
#include <map>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  map<char,string> mymap;

  mymap['a']="an element";
  mymap['b']="another element";
  mymap['c']=mymap['b'];

  cout << "mymap['a'] is " << mymap['a'] << endl;
  cout << "mymap['b'] is " << mymap['b'] << endl;
  cout << "mymap['c'] is " << mymap['c'] << endl;
  cout << "mymap['d'] is " << mymap['d'] << endl;
  
  assert(mymap['a']=="an element");
  assert(mymap['b']=="another element");
  assert(mymap['c']!=mymap['b']);
  assert(mymap['d']!=string());

  cout << "mymap now contains " << (int) mymap.size() << " elements." << endl;

  return 0;
}
