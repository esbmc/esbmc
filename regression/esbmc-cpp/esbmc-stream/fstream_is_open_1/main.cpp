// fstream::is_open
#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;

int main () {

  fstream filestr;
  filestr.open ("test");
  assert(filestr.is_open());
  if (filestr.is_open())
  {
    filestr << "File successfully open";
    filestr.close();
    assert(!(filestr.is_open()));
  }
  else
  {
    cout << "Error opening file";
  }
  return 0;
}
