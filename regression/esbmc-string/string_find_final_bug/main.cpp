//TEST FAILS
// string::find
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("There.");
  string str2 ("needle");
  size_t found;

  found=str.find('.');
  assert(int(found)==51);
  if (found!=string::npos)
    cout << "Period found at: " << int(found) << endl;

  return 0;
}
