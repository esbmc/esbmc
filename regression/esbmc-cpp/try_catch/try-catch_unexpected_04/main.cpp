// myfunction()'s throw(int) spec is violated by `throw 'x'` (char), so
// std::unexpected runs myunexpected, which throws 5 (int) — permitted by the
// spec, so it propagates and is caught by catch(int). The program terminates
// normally: VERIFICATION SUCCESSFUL (confirmed against g++ -std=c++11, exit 0).
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

