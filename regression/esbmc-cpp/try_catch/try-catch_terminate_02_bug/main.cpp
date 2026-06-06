// set_terminate example. The uncaught `throw 0` invokes std::terminate, which
// (under --lower-exceptions, routed through the OM) runs the installed handler
// myterminate. It calls abort(), which ESBMC models as a silent, valid
// termination (assume(0), not a property violation) — so the program verifies
// SUCCESSFUL. The lowering honours the custom handler; the imperative path
// asserted at the terminate point without running it.
#include <iostream>
#include <exception>
#include <cstdlib>
using namespace std;

void myterminate () {
  cerr << "terminate handler called\n";
  abort();  // forces abnormal termination
}

int main (void) {
  set_terminate (myterminate);
  throw 0;  // unhandled exception: calls terminate handler
  return 0;
}
