/*
 * dynamic cast, check address
 */
#include <cassert>
using namespace std;

struct A {
  virtual ~A() { };
};

struct B : A { };

int main() {
  B bobj;
  A* ap = &bobj;
  void * vp = dynamic_cast<void *>(ap);
  assert(vp != &bobj); // FAIL, should be the same address
}

