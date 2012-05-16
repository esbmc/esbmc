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


  // Using iterators:                      0123456789*123456789*
  string::iterator it = str.begin();   //  ^
  str.replace(it,str.end()-3,str3);    // "sample phrase!!!"
  assert(str == "sample phrase!!!");
  
  cout << str << endl;
  return 0;
}
