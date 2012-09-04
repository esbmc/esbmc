// istream::unget example
#include <iostream>
#include <cassert>
using namespace std;

int main () {
  char c;
  int n;
  char str[256];

  cout << "Enter a number or a word: ";
  c = cin.get();

  if ( (c >= '0') && (c <= '9') )
  {
    cin.unget();
    assert((int)cin.gcount() == 0);
    cin >> n;
    cout << "You have entered number " << n << endl;
  }
  else
  {
    cin.unget();
    assert((int)cin.gcount() == 0);
    cin >> str;
    cout << " You have entered word " << str << endl;
  }

  return 0;
}
