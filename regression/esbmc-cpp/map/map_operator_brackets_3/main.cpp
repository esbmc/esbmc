// accessing mapped values
#include <map>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  map<char,string> mymap;
  mymap['a']="abc";
  assert(mymap['a']=="abc");
  return 0;
}
