#include <iostream>
#include <exception>
using namespace std;

void myunexpected () {
  cerr << "unexpected handler called\n";
  throw;
}

void myfunction () throw () {
  throw 'x';
}

int main (void) {
  set_unexpected (myunexpected);
  try {
    myfunction();
  }
  catch (int) { cerr << "caught int\n"; }
  catch (bad_exception be) { cerr << "caught bad_exception\n"; }
  catch (...) { cerr << "caught other exception (non-compliant compiler?)\n"; }
  return 0;
}
