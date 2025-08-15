#include <assert.h>

struct Counter {
  int &ref;
  Counter(int &r) : ref(r) {}
  void increment() {ref++; }
};

int main() {
   int x = 0;
   Counter c(x);
   c.increment();
   c.ref++;
   assert(x == 2);
   return 0;
} 
