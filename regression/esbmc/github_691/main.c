#include <assert.h>

#define ARR_SIZE 1

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

typedef union {
  uint32_t dwords[2];
  uint8_t bytes[8];
} qword_t;

uint32_t nondet_uint32_t() {
  uint32_t val;
  return val;
}

int main() {

  qword_t a[ARR_SIZE];
  qword_t b[ARR_SIZE];
  
  for (int i = 0; i < ARR_SIZE; i++) { 
    for (int j = 0; j < 2; j++) {
      a[i].dwords[j] = nondet_uint32_t();
    }
    b[i] = a[i];
  }

  assert(b[0].bytes[0] == 0);

  return 0;
}
