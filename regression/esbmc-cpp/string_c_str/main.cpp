// strings and c-strings
//Example from C++ reference, avaliable at http://www.cplusplus.com/reference/string/string/c_str/
#include <iostream>
#include <cstring>
#include <string>
using namespace std;

int main ()
{
  char * cstr, *p;

  string str ("Please split this phrase into tokens");

  cstr = new char [str.size()+1];
  strcpy (cstr, str.c_str());

  // cstr now contains a c-string copy of str

  p=strtok (cstr," ");
  while (p!=NULL)
  {
    cout << p << endl;
    p=strtok(NULL," ");
  }

  delete[] cstr;  
  return 0;
}
