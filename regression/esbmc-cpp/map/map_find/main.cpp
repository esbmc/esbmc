// map::find
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::iterator it;

  mymap['a']=50;
  mymap['b']=100;
  mymap['c']=150;
  mymap['d']=200;

  assert(mymap.find('a')->first == 'a');
  assert(mymap.find('a')->second == 50);
  
  assert(mymap.find('b')->first == 'b');
  assert(mymap.find('b')->second == 100);

  assert(mymap.find('c')->first == 'c');
  assert(mymap.find('c')->second == 150);
  
  assert(mymap.find('d')->first == 'd');
  assert(mymap.find('d')->second == 200);

  it=mymap.find('b');
  mymap.erase (it);
  mymap.erase (mymap.find('d'));

  // print content:
  cout << "elements in mymap:" << endl;
  cout << "a => " << mymap.find('a')->second << endl;
  cout << "c => " << mymap.find('c')->second << endl;

  return 0;
}
