// appending to string
//Example from C++ Reference, avaliable at http://www.cplusplus.com/reference/string/string/append/

#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str;
  string str2="Writing ";
  string str3="print 10 and then 5 more";

  // used in the same order as described above:
  str.append(str2);                       // "Writing "
  str.append(str3,6,3);                   // "10 "
  str.append("dots are cool",5);          // "dots "
  str.append("here: ");                   // "here: "
  str.append(10,'.');                     // ".........."
  str.append(str3.begin()+8,str3.end());  // " and then 5 more"
  str.append<int>(5,0x2E);                // "....."

  assert(str == "Writing 10 dots here: .......... and then 5 more.....");
  
  cout << str << endl;
  return 0;
}
