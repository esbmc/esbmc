// map::insert
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> mymap;
  map<char,int>::iterator it;
  char chararray[4] = {'a', 'b', 'c', 'z'};
  int intarray[4] = {100, 300, 400, 200};

  mymap['a'] = 100;
  mymap['z'] = 200;

  // second insert function version (with hint position):
  it=mymap.begin();
  mymap.insert (it, pair<char,int>('b',300));  // max efficiency inserting
  mymap.insert (it, pair<char,int>('c',400));  // no max efficiency inserting
  int i = 0;
 cout << "mymap contains:\n";
  for ( it=mymap.begin() ; it != mymap.end(); it++)
  {
    cout << (*it).first << " => " << (*it).second << endl;
    assert( (*it).first == chararray[i]);
    assert( (*it).second == intarray[i]);
    i++;
      }
    


  return 0;
}
