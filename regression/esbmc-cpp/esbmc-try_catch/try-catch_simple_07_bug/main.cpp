#include <exception>
#include <cassert>

int main (void) {
  try {
    throw 'a';   // throws char
  }
  catch (int) { return 1; }
  catch (char) { assert(0); return 2; }
  return 0;
}
