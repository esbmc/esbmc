#include <string.h>
#include <assert.h>

int main()
{
	char char_array[4] = { 0x1, 0x2, 0x3, 0x4 };
	short short_array[2];
	memcpy(short_array, char_array, 4);
	short s0 = short_array[0];
	short s1 = short_array[1];
  /* little-endian */
	assert(s0 == 0x0201);
	assert(s1 == 0x0403);
}

