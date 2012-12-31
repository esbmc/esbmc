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
  
cout << found << endl;
  
  return 0;
}
