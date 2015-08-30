#include <exception>
using namespace std;

void myfunction () throw (int,bad_exception) {
  throw 5;
}

void myfunction1 () throw (char) {
  throw 5;
}

int main (void) {
  try {
    try {
      myfunction();
    } catch(...) {}

    myfunction1();
  }
  catch (int) { return 1; }
  catch (bad_exception be) { return 2; }
  return 0;
}

