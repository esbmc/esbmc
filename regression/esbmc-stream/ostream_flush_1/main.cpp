// Flushing files
#include <fstream>
#include <cassert>
using namespace std;

int main () {

  ofstream outfile ("test");
  assert(outfile.is_open());
  for (int n=0; n<100; n++)
  {
    outfile << n;
    outfile.flush();
  }

  outfile.close();

  return 0;
}
