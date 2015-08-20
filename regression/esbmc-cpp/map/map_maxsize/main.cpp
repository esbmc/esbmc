// map::max_size
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  int i;
  map<int,int> mymap;

  if (mymap.max_size()>1000)
  {
    for (i=0; i<1000; i++) mymap[i]=0;
    cout << "The map contains 1000 elements.\n";
  }
  else cout << "The map could not hold 1000 elements.\n";

  return 0;
}
