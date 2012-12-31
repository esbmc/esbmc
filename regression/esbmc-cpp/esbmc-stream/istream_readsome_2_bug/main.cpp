//TEST FAILS
// read a file into memory
#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;

int main() {
  int length;
  char * buffer;

  ifstream is;
  is.open ("test", ios::binary );

  // get length of file:
  is.seekg (0, ios::end);
  length = is.tellg();
  is.seekg (0, ios::beg);

  // allocate memory:
  buffer = new char [length];

  // read data as a block:
  is.readsome (buffer,length/2);
  assert(!(length/2 == (int)is.gcount()));
  is.close();

  cout.write (buffer,length);

  delete[] buffer;
  return 0;
}
