#include <assert.h>
#include <uchar.h>

int main()
{
	const char16_t *s = u"\x1234";
	assert(s[0] == u'\x1234');
	assert(s[1] == u'\0');
}
