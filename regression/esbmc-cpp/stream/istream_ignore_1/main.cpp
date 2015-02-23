// istream ignore
#include <iostream>
#include <cassert>
using namespace std;
  
int main () {
  char first, last;
  int i, j;
  cout << "Enter your first and last names: ";
  
  first=cin.get();
  assert((int)cin.gcount() == 1);
  cin.ignore(256,' ');
  
  last=cin.get();
  assert((int)cin.gcount() == 1);
  cout << "Your initials are " << first << last;
  
  return 0;
}
