#include <assert.h>

#define test_type int
#define type_size sizeof(test_type)
#define array_size 5 // This must be a constant


void manual_memset(void *ptr, int byte, unsigned N)
{
  char *P = ptr;
  for(unsigned i = 0; i < N; i++)
    P[i] = (unsigned char) byte;
}

void test_memset(void *actual, void *expected, int byte, unsigned num_of_bytes)
{  
  memset(actual, byte, num_of_bytes);
  manual_memset(expected, byte, num_of_bytes);

  char *ptr1 = actual;
  char *ptr2 = expected;
  for(int i = 0; i < num_of_bytes; i++)
    assert(ptr1[i] == ptr2[i]);
}

void test_nondet_complete()
{
  test_type arr[array_size];
  test_type copy[array_size];
  for(int i = 0; i < array_size; i++)
    arr[i] = copy[i];

  int byte;
  unsigned num_of_bytes = type_size * array_size;

  // Try to initialize any number of bytes
  for(int i = 0; i <= num_of_bytes; i++)
  {
    test_memset(arr, copy, byte, num_of_bytes-i);   
  }

  // Try to initialize any number of bytes with an offset
  for(int i = 0; i < num_of_bytes; i++)
  {
    test_memset(((char*)arr)+i, ((char*)copy)+i, byte, num_of_bytes-i);   
  }
}

int main()
{
    test_nondet_complete();
}