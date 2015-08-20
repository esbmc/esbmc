#include <cassert>

#define size_t int

void *memcpy(void *dst, const void *src, size_t n) {
	__ESBMC_HIDE:
        char *cdst = static_cast<char *>(dst);
        const char *csrc = static_cast<const char *>(src);
        for (size_t i = 0; i < n; i++)
		cdst[i] = csrc[i];
	return dst;
}

int main ()
{
   const char src[3] = "ht";
   char dest[3];

   memcpy(dest, src, 3);

   assert(dest[1]=='t');
   
   return 0;
}
