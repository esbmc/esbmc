// fstream::open
#include <fstream>
#include <cassert>
using namespace std;

int main () {

  fstream filestr;

  filestr.open ("test", fstream::in | fstream::out | fstream::app);
  assert(filestr.is_open());
  // >> i/o operations here <<

  filestr.close();

  return 0;
}
