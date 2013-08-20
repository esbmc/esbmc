// ostream constructor
#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;

int main () {
  filebuf fb;
  fb.open ("test",ios::out);
  ostream os(&fb);
//  assert(os.is_open());
  os << "Test sentence\n";
  
  fb.close();
//  assert(!(os.is_open()));
  return 0;
}
