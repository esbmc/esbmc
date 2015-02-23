#include <exception>
#include <cassert>
using namespace std;

void myfunction () throw (int,bad_exception) {
  throw 5;
}

void myfunction1 () throw (char) {
  throw 'x';
}

int main (void) {
  try {
    myfunction();

    try {
      myfunction1();
    }
    catch (char) { throw 'x'; }
  }
  catch (int) { return 1; }
  catch (bad_exception be) { return 2; }
  catch (char) { assert(0); }
  return 0;
}
