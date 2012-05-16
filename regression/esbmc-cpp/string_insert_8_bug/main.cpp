//TEST FAILS

// inserting into a string
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str="to be question";
  string str2="the ";
  string str3="or not to be";
  string::iterator it;

  // used in the same order as described above:
  str.insert(6,str2);                 // to be (the )question
  assert(str != "to be the question");
  
  cout << str << endl;
  return 0;
}
