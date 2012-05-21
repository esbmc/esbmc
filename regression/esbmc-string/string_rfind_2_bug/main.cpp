//TEST FAILS
// string::rfind
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("The sixth sick sheik's sixth sheep's sick.");
  string key ("sixth");
  size_t found;

  found=str.rfind(key);
  
  
  if (found!=string::npos)
    str.replace (found,key.length(),"seventh");
	assert(str != "The sixth sick sheik's seventh sheep's sick.");
  cout << str << endl;

  return 0;
}
