// string::rfind
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("T six sd six shck.");
  string key ("six");
  size_t found;

  found=str.rfind(key);

assert(found == 9);
  
cout << found << endl;
  
//  if (found!=string::npos)
//    str.replace (found,key.length(),"seventh");
//	assert(str == "The sixth sick sheik's seventh sheep's sick.");
//  cout << str << endl;

  return 0;
}
