#include <exception>
using namespace std;

void myfunction1 () throw (char) {
  throw 'x';
}

void myfunction () throw (int,bad_exception) {
  myfunction1();
  throw 5;
}

int main (void) {
  try {
    myfunction();
  }
  catch (int) { return 1; }
  catch (bad_exception be) { return 2; }
  return 0;
}
