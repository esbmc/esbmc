//TEST FAILS
// istream constructor
#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;

int main () {
  filebuf fb;
  fb.open ("test",ios::in);
  assert(!fb.is_open());
  istream is(&fb);
  cout << char(is.get());
  fb.close();
  return 0;
}
