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

  // Using iterators:                      0123456789*123456789*
  string::iterator it = str.begin();   //  ^
  str.replace(it,str.end()-3,str3);    // "sample phrase!!!"
  assert(str != "sample phrase!!!");
  
  str.replace(it,it+6,"replace it",7); // "replace phrase!!!"
  assert(str != "replace phrase!!!");
  
  it+=8;                               //          ^
  str.replace(it,it+6,"is cool");      // "replace is cool!!!"
  assert(str != "replace is cool!!!");  
  
  cout << str << endl;
  return 0;
}
