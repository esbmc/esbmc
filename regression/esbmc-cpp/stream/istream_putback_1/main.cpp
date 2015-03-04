// istream putback
#include <iostream>
#include <cassert>
using namespace std;

int main () {
  char c;
  int n;
  char str[256];

  cout << "Enter a number or a word: ";
  c = cin.get();
      assert((int)cin.gcount() == 1);
  if ( (c >= '0') && (c <= '9') )
  {
    cin.putback (c);
    assert((int)cin.gcount() == 0);
    cin >> n;
    cout << "You have entered number " << n << endl;
  }
  else
  {
    cin.putback (c);
    assert((int)cin.gcount() == 0);
    cin >> str;
    cout << " You have entered word " << str << endl;
  }

  return 0;
}
