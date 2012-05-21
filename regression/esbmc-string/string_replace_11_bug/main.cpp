//TEST FAILS
// replacing in a string
#include <iostream>
#include <string>
#include <cassert>

using namespace std;

int main ()
{
  string base="this is a test string.";
  string str2="n example";
  string str3="sample phrase";
  string str4="useful.";

  // function versions used in the same order as described above:

  // Using positions:                 0123456789*123456789*12345
  string str=base;                // "this is a test string."
  str.replace(9,5,str2);          // "this is an example string."
  assert(str != "this is an example string.");
  
  
  cout << str << endl;
  return 0;
}
