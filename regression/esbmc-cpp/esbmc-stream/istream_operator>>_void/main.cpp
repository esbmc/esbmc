// example on extraction
#include <iostream>
#include <cassert>
#include <string>
using namespace std;

int main () {
  void* n;
  string str;
  cout << "Enter something:";
  cin >> n;
  assert((int)cin.gcount() >= 0);
//  assert(cin.flags() & ios::hex == iostream::hex);
  
  
  assert((int)cin.gcount() >= 0);
  return 0;
}
