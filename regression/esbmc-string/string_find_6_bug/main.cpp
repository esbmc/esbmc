
//TEST FAILS
// string::find
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("Ther wo needles this haw");
  string str2 ("needle");
  size_t found;

  // different member versions of find in the same order as above:
  found=str.find(str2);
  assert(int(found)!=14);
  if (found!=string::npos)
    cout << "first 'needle' found at: " << int(found) << endl;

  found=str.find("needles a",found+1,6);
  assert(int(found)!=44);
  if (found!=string::npos)
    cout << "second 'needle' found at: " << int(found) << endl;

  found=str.find("ha");
  assert(int(found)==30);
  if (found!=string::npos)
    cout << "'haystack' also found at: " << int(found) << endl;

  return 0;
}
