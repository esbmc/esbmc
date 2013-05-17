// set_unexpected example
#include <iostream>
#include <exception>
using namespace std;

void myfunction () {
  throw 1;
}

int main (void) {
  try {
    myfunction();
  }
  catch (int) { cerr << "caught int\n"; }
  return 0;
}
