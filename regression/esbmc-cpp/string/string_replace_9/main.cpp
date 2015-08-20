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


  string str=base;/*                // "this is a test string."
  str.replace(9,5,str2);          // "this is an example string."
  assert(str == "this is an example string.");
  
  str.replace(19,6,str3,7,6);     // "this is an example phrase."
  assert(str == "this is an example phrase.");
  
  str.replace(8,10,"just all",6); // "this is just a phrase."
  assert(str == "this is just a phrase.");
  
  str.replace(8,6,"a short");     // "this is a short phrase."
  assert(str == "this is a short phrase.");
  
  str.replace(22,1,3,'!');        // "this is a short phrase!!!"
  assert(str == "this is a short phrase!!!");
  */
  // Using iterators:                      0123456789*123456789*
  string::iterator it = str.begin();   //  ^
  str.replace(it,str.end()-3,str3);    // "sample phrase!!!"
  cout << str << endl;
  assert(str == "sample phraseng.");
  
  str.replace(it,it+6,"replace it",7); // "replace phrase!!!"
  cout << str << endl;
  assert(str == "replace phraseng.");
  
  it += 8;                               //          ^
  str.replace(it,it+6,"is cool");      // "replace is cool!!!"
  cout << str << endl;
  assert(str == "replace is coolng.");  
//  assert(str == "is cool phraseng.");
  
  str.replace(it+4,str.end()-4,4,'o'); // "replace is cooool!!!"
  cout << str << endl;
  assert(str == "replace is coooolng.");
  
  cout << str << endl;
  return 0;
}
