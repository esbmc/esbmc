// using fstream constructors.
#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;

int main () {

  fstream filestr ("test", fstream::in | fstream::out);

  assert(filestr.is_open());
  // >> i/o operations here <<

  filestr.close();

  return 0;
}
