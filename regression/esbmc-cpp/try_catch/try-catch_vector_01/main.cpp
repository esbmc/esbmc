// length_error example
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <vector>
using namespace std;

int main (void) {
  try {
    vector<int> myvector;
    myvector.resize(myvector.max_size()+1);
  }
  catch (length_error& le) {
    cerr << "Length error: " << le.what() << endl;
  }
  return 0;
}
