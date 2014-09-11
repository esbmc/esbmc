// map::size
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  mymap['a']=101;
  mymap['b']=202;
  mymap['c']=302;
  assert(mymap.size() == 3);
  cout << "mymap.size() is " << (int) mymap.size() << endl;

  return 0;
}
