// string::substr
//Example from c++ reference, avaliable at www.cplusplus.com/reference/string/string/substr/
//TEST FAILS

#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str="We ge, live.";
                             // quoting Alfred N. Whitehead
  string str2, str3;
  int pos;

  str2 = str.substr (3,2); // "generalities"

  pos = str.find("live");    // position of "live" in str
  str3 = str.substr (pos);   // get from "live" to the end

  cout << str2 << ' ' << str3 << endl;
  assert((str2 == "generalities")&&(str3 != "live in details"));
  return 0;
}
