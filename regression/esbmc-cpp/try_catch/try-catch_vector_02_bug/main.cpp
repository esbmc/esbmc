// out_of_range example: vector::at(20) on a size-10 vector throws
// std::out_of_range, which the catch handles, so the program completes normally
// (VERIFICATION SUCCESSFUL). vector::at is modelled to throw, per the C++
// standard, rather than asserting like operator[].
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <vector>
using namespace std;

int main (void) {
  vector<int> myvector(10);
  try {
    myvector.at(20)=100;      // vector::at throws an out-of-range
  }
  catch (out_of_range& oor) {
    cerr << "Out of Range error: " << oor.what() << endl;
  }
  return 0;
}
