#include <cassert>

int main (void) {
  try {
    throw 5;
    goto ab;
  }
  catch (int) { assert(0); }

ab: return 0;
}

