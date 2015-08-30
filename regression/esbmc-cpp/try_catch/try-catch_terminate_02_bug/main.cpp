// set_terminate example
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
