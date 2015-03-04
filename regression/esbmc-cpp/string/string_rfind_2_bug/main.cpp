//TEST FAILS
// string::rfind
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("T six sd six hck.");
  string key ("six");
  size_t found;

  found=str.rfind(key);

assert(found == 9);
  
  
  if (found!=string::npos)
    str.replace (found,key.length(),"se");
	assert(str != "T six sd se hck.");
  cout << str << endl;

  return 0;
}
