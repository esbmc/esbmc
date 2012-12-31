//Example from C++ reference, avaliable at www.cplusplus.com/reference/string/string/substr/
// string::substr
#include <iostream>
#include <string>
using namespace std;

int main ()
{
  string str="We think in generalities, but we live in details.";
                             // quoting Alfred N. Whitehead
  string str2, str3;
  size_t pos;

  str2 = str.substr (12,12); // "generalities"

  pos = str.find("live");    // position of "live" in str
  str3 = str.substr (pos);   // get from "live" to the end

  cout << str2 << ' ' << str3 << endl;

  return 0;
}
