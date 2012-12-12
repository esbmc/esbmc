// istream peek
#include <iostream>
#include <cassert>
#include <cstring>
using namespace std;

int main () {
  char c;
  int n;
  char str[256];

  cout << "Enter a number or a word: ";
  c=cin.peek();
  
  if ( (c >= '0') && (c <= '9') )
  {
    cin >> n;
    cout << "You have entered number " << n << endl;
  }
  else
  {
    cin >> str;
    cout << " You have entered word " << str << endl;
    
  }
  assert((int)cin.gcount() >= 0 && (int)cin.gcount() < 256);
  
  return 0;
}
