// map::insert
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::iterator it;

  mymap['a'] = 100;
  mymap['b'] = 300;
  mymap['c'] = 400;
  mymap['z'] = 200;
  
  char chararray[4] = {'a', 'b', 'c', 'z'};
  int intarray[4] = {100, 300, 400, 200};
  
  // third insert function version (range insertion):
  map<char,int> anothermap;
  anothermap.insert(mymap.begin(),mymap.find('c'));
  int i = 0;
  // showing contents:
  cout << "mymap contains:\n";
  for ( it=mymap.begin() ; it != mymap.end(); it++ ){
    cout << (*it).first << " => " << (*it).second << endl;
    assert((*it).first == chararray[i]);
    assert((*it).second == intarray[i]);
    i++;
    }
  i = 0;
  cout << "anothermap contains:\n";
  for ( it=anothermap.begin() ; it != anothermap.end(); it++ ){
    cout << (*it).first << " => " << (*it).second << endl;
    assert((*it).first == chararray[i]);
    assert((*it).second != intarray[i]);
    i++;
    }

  return 0;
}
