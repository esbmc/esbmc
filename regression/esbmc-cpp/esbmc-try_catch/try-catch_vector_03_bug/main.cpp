// out_of_range example
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <vector>
using namespace std;

int main (void) {
  vector<int> myvector(10);
  try {
    myvector.at(0)=1;
    myvector.at(20)=100;      // vector::at throws an out-of-range
  }
  catch (out_of_range& oor) {
    assert(myvector.at(0)==2);
    cerr << "Out of Range error: " << oor.what() << endl;
  }
  return 0;
}
