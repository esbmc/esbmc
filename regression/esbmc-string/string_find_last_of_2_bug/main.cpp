//TEST FAILS
// string::find_last_of
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1 ("/usr/bin/man");
  string str2 ("c:\\windows\\winhelp.exe");
  
  size_t found;
  
  cout << "Splitting: " << str1 << endl;
  found=str1.find_last_of("/\\");
  assert(int(found) != 8);
  cout << " folder: " << str1.substr(0,found) << endl;
  cout << " file: " << str1.substr(found+1) << endl;
  
  cout << "Splitting: " << str2 << endl;
  found=str2.find_last_of("/\\");
  assert(int(found) != 10);
  cout << " folder: " << str2.substr(0,found) << endl;
  cout << " file: " << str2.substr(found+1) << endl;
  
  
  return 0;
}
