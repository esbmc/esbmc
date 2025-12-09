#include <assert.h>
#include <uchar.h>

int main()
{
	const char32_t *s = U"\x12345";
	char32_t c0 = s[0];
	char32_t c1 = s[1];
	assert(s[0] == U'\x12345');
	assert(s[1] == U'\0');
}
