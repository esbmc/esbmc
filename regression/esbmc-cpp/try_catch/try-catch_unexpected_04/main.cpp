// myfunction's throw(int) spec is violated by `throw 'x'` (char), which runs
// myunexpected(); it throws 5, an int, which the spec permits, so it
// propagates and is caught by catch(int). Ground-truthed against real
// clang++ -std=c++03: prints "unexpected called" / "caught int", exits 0 --
// a well-defined recovery, not a violation (github #6022).
#include <iostream>
#include <exception>
using namespace std;

void myunexpected () {
  cerr << "unexpected called\n";
  throw 5;
}

void myfunction () throw (int) {
  throw 'x';
}

int main (void) {
  set_unexpected (myunexpected);
  try {
    myfunction();
  }
  catch (int) { cerr << "caught int\n"; }
  catch (...) { cerr << "caught other exception (non-compliant compiler?)\n"; }
  return 0;
}

