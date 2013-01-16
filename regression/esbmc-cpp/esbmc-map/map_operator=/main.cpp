// assignment operator with maps
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  map<char,int> first;
  map<char,int> second;

  first['x']=8;
  first['y']=16;
  first['z']=32;

  second=first;           // second now contains 3 ints
  assert(second['x'] == 8);
  assert(second['y'] == 16);
  assert(second['z'] == 32);
  
  first=map<char,int>();  // and first is now empty
  assert(first.size() == 0);

  cout << "Size of first: " << int (first.size()) << endl;
  cout << "Size of second: " << int (second.size()) << endl;
  return 0;
}
