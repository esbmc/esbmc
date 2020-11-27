#include <stdint.h>

#define MAX_SIZE_IN_BYTES 1000

typedef uint8_t UINT8;
typedef uint16_t UINT16;
typedef uint32_t UINT32;
typedef uint64_t UINT64;

UINT8 nondet_uint8() {UINT8 val; return val;}
UINT16 nondet_uint16() {UINT16 val; return val;}
UINT32 nondet_uint32() {UINT32 val; return val;}
UINT64 nondet_uint64() {UINT64 val; return val;}

typedef struct {
  UINT64           address;            
  UINT32           size;               
  UINT16           version;            
  UINT8            checksum;           
} struct_t;

typedef union {
  UINT8 bytes[MAX_SIZE_IN_BYTES + 0x100];
} my_union_t;

typedef union {
    struct_t just_placeholder;
    UINT8 bytes[MAX_SIZE_IN_BYTES + 0x300];
} my_buffer_union_t;

my_union_t MyBuffer;

#define MyBufferPtr ((my_buffer_union_t *) MyBuffer.bytes)

int main() {
  struct_t * my_struct = (struct_t *) malloc(sizeof(struct_t));
  my_struct ->address = nondet_uint64();
  my_struct ->size = nondet_uint32();
  my_struct ->version = nondet_uint16();
  my_struct ->checksum = nondet_uint8();

  if (MyBufferPtr->bytes == (UINT8 *) my_struct ->address)
    return 1;

  if (MyBufferPtr->bytes < (UINT8 *) my_struct ->address)
    return 1;

  return 0;
}
