/* memchr example */
#include <iostream>
using std::cout;
using std::cin;
using std::endl;
#include <cstring>

int main ()
{
  char * pch;
  char str[] = "Example string";
  pch = (char*) memchr (str, 'p', 14);
  if (pch!=NULL)
    cout << "'p' found at position " << pch-str+1 << "." << endl;
  else
    cout << "'p' not found." << endl;
  return 0;
}
