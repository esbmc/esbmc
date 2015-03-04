#include <exception>
using namespace std;

void myfunction1 () throw (char) {
  throw 'x';
}

void myfunction () throw (int,bad_exception) {
  try {
    myfunction1();
  }
  catch(...) { throw 5; }
  throw 5;
}

int main (void) {
  try {
    myfunction();
    myfunction1();
  }
  catch (int) { return 1; }
  catch (bad_exception be) { return 2; }
  return 0;
}
