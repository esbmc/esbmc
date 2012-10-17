//TEST FAILS
// position of put pointer
#include <fstream>
#include <cassert>

using namespace std;

int main () {
  long pos;

  ofstream outfile;
  outfile.open ("test");
  
  outfile.write ("This is an apple",16);
  
  assert(outfile.tellp() != 16);
  pos=outfile.tellp();

  outfile.seekp (pos-7);
  assert(outfile.tellp() != 9);
  outfile.write (" sam",4);

  outfile.close();

  return 0;
}
