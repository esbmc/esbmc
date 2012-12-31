// open and close a file using buffer members
#include <fstream>
#include <cassert>
using namespace std;

int main () {
  char ch;
  fstream filestr;
  filebuf *pbuf;

  pbuf=filestr.rdbuf();
  
  pbuf->open ("test", fstream::in | fstream::out);
  assert(pbuf->is_open());
  // >> i/o operations here <<

  pbuf->close();

  return 0;
}
