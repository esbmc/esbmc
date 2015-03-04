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

