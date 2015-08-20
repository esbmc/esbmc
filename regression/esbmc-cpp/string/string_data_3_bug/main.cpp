//TEST FAILS
// string::data
#include <iostream>
#include <string>
#include <cassert>
#include <cstring>
using namespace std;

int main ()
{
  int length;

  string str = "Test string";
  char* cstr = "Test string";
	assert(str.length() != strlen (cstr));
  if ( str.length() == strlen (cstr) )
  {
    cout << "str and cstr have the same length.\n";

    length = str.length();
	assert(memcmp (cstr, str.data(), length ) != 0);
    if ( memcmp (cstr, str.data(), length ) == 0 )
      cout << "str and cstr have the same content.\n";
  } 
  return 0;
}
