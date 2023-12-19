#include <assert.h>
#include <wchar.h>

int main()
{
	wchar_t s[] = L"0123";
	size_t n = 5;
	assert(sizeof(s) == sizeof(*s) * n);
	for (size_t i=0; i<n-1; i++) {
		wchar_t v = s[i];
		wchar_t w = L'0' + i;
		assert(v == w);
	}
}
