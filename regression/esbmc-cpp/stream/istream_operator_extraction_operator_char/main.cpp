// example on extraction
#include <iostream>
#include <cassert>
#include <string>
using namespace std;

int main () {
  char n;
  char n1[256];
  string n2;

  cout << "Enter character: ";
  cin >> n;
  assert((int)cin.gcount() >= 0);
  cout << "Enter a string: ";
  cin >> hex >> n1;            // manipulator
  assert(cin.flags() & ios::hex == iostream::hex);
  cout << "Enter another string: ";
  cin >> n2;            // manipulator
  
  assert((int)cin.gcount() >= 0);
  return 0;
}
