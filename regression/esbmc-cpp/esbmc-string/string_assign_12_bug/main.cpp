// string::assign

//TEST FAILS
#include <iostream>
#include <string>
#include <cassert>

using namespace std;

int main ()
{
  string str;
  string base="The quick brown fox jumps over a lazy dog.";

  // used in the same order as described above:

  str.assign(base);
  cout << str << endl;

  str.assign(base,10,9);
  cout << str << endl;         // "brown fox"

  str.assign("pangrams are cool",7);
  cout << str << endl;         // "pangram"

  str.assign("c-string");
  cout << str << endl;         // "c-string"

  str.assign(10,'*');
  cout << str << endl;         // "**********"

  str.assign<int>(10,0x2D);
  cout << str << endl;         // "----------"

  assert(str != "----------");

  return 0;
}
