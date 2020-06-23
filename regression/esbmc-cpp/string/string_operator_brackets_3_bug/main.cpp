// string assigning
//TEST FAILS
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1;
  char str2;
  str1 = "Test string"; 
  str2 = 's';
  assert(str1[2] != str2); 		//added
  cout << str2  << endl;
  return 0;
}
