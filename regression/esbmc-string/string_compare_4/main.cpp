// comparing apples with apples
//Example from C++ reference, avaliable at http://www.cplusplus.com/reference/string/string/compare/

#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1 ("green apple");
  string str2 ("red apple");

  assert(str1.compare(str2) != 0);
  if (str1.compare(str2) != 0)
    cout << str1 << " is not " << str2 << "\n";

  assert(str1.compare(6,5,"apple") == 0);  
  if (str1.compare(6,5,"apple") == 0)
    cout << "still, " << str1 << " is an apple\n";

  assert(str2.compare(str2.size()-5,5,"apple") == 0);
  if (str2.compare(str2.size()-5,5,"apple") == 0)
    cout << "and " << str2 << " is also an apple\n";

  assert(str1.compare(6,5,str2,4,5) == 0);
  if (str1.compare(6,5,str2,4,5) == 0)
    cout << "therefore, both are apples\n";


//asserts added by Hendrio
  return 0;
}
