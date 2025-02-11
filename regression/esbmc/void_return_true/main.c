#include <assert.h>
int i;
void func() {
  return ((void)(i++));
}

int main() {
  func();
  assert(i == 1);
  return 0;
}