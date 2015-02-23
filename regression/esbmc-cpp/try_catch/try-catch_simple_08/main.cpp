#include <exception>
#include <cassert>

int main (void) {
  try {
    throw 1;
  }
  catch (char) { assert(0); return 2; }
  catch (int) { return 1; }
  return 0;
}
