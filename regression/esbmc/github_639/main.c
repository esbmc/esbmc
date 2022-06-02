#include <assert.h>

union my_obj {
  char a;
  int b;
  int c;
};

#define test_type union my_obj
#define test_size sizeof(union my_obj)

void manual_memset(void *ptr, int byte, unsigned N)
{
  char *dst = ptr;
  for(unsigned i = 0; i < N; i++)
    dst[i] = (unsigned char) byte;
}

void test_memset(test_type initial, int byte, unsigned N)
{ 
  test_type actual = initial;
  test_type expected = initial;
  memset(&actual, byte, N);
  manual_memset(&expected, byte, N);

  // byte by byte comparation then
  char *ptr1 = &actual;
  char *ptr2 = &expected;
  for(int i = 0; i < N; i++)
    assert(ptr1[i] == ptr2[i]); 
}
int main()
{
    int byte = 2;
    union my_obj test;
    test.b = 0;
    test_memset(test, byte, test_size);
}
