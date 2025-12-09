// stringstream::str
#include <sstream>
#include <cassert>
using namespace std;

int main () {

  stringstream oss;
  float val = 70.23;
  
  oss << val;
  assert(oss.str() == "70.23");
  
  return 0;
}
