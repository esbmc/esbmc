// accessing mapped values
#include <map>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  map<char,string> mymap;
  mymap['a']="element";
  assert(mymap['a']=="element");
  return 0;
}
