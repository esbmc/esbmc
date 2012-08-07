#include <exception>
using namespace std;

void myunexpected () {
  throw;
}

void myfunction () throw (int e, bad_exception) {
  throw 'x';
}

int main (void) {
  try {
    throw 1;
    myfunction();
  }
  catch (int) { return 3; }
  catch (bad_exception be) { return 2; }
  catch (...) { return 1; }
  return 0;
}
