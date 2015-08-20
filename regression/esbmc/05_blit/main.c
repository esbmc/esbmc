#include "common.h"
unsigned long src[1024] =
{
  0x00005678, 0x12340000, 0x02040608, 0x00000001,
  0x12345678, 0x12345678, 0x12345678, 0x12345678,
  0x00005678, 0x12340000, 0x02040608, 0x00000001,
  0x12345678, 0x12345678, 0x12345678, 0x12345678,
  0x00005678, 0x12340000, 0x02040608, 0x00000001,
  0x12345678, 0x12345678, 0x12345678, 0x12345678,
  0x00005678, 0x12340000, 0x02040608, 0x00000001,
  0x12345678, 0x12345678, 0x12345678, 0x12345678,
  0x00005678, 0x12340000, 0x02040608, 0x00000001,
  0x12345678, 0x12345678, 0x12345678, 0x12345678,
  0x00005678, 0x12340000, 0x02040608, 0x00000001,
  0x10101010, 0x12345678, 0x10101010, 0x12345678,
  0x00005678, 0x12340000, 0x02040608, 0x00000001,
  0x10101010, 0x12345678, 0x10101010, 0x12345678,
  0x00005678, 0x12340000, 0x02040608, 0x00000001,
  0x10101010, 0x12345678, 0x10101010, 0x12345678,
  0
};
unsigned long dst[1024];
void blit(unsigned long saddr, unsigned long daddr, unsigned long n)
{
  int d1, d2;
  unsigned long *s, *d, x, y;
  unsigned long soff, doff;
  s = (unsigned long *) ((unsigned long) src + (saddr >> 5));
  d = (unsigned long *) ((unsigned long) dst + (daddr >> 5));
  soff = saddr & 0x1f;
  doff = daddr & 0x1f;
  if (soff > doff)
    {
      d1 = soff - doff;
      d2 = 32 - d1;
      y = *d & (0xffffffff << (32 - doff));
      x = *s++ & ((unsigned) 0xffffffff >> soff);
      y |= x << d1;
      for (; n >= 32; n -= 32, s++, d++)
	{
	  x = *s;
	  *d = y | (x >> d2);
	  y = x << d1;
	}
      x = *d & (0xffffffff << (32 - n + d1));
      *d = y | x;
    } 
  else if (soff < doff)
    {
      d1 = doff - soff;
      d2 = 32 - d1;
      y = *d & (0xffffffff << (32 - doff));
      x = *s++ & ((unsigned) 0xffffffff >> soff);
      for (; n >= 32; n -= 32, s++, d++)
	{
	  *d = y | (x >> d1);
	  y = x << d2;
	  x = *s;
	}
      x = *d & (0xffffffff << (32 - n + d1));
      *d = y | (x >> d1);
    } 
  else
    {
      if (soff)
	{
	  y = *d & (0xffffffff << (32 - doff));
	  x = *s++ & ((unsigned) 0xffffffff >> soff);
	  *d++ = x | y;
	  n -= (32 - soff);
	}
      for (; n >= 32; n -= 32, s++, d++)
	*d = *s;
      if (n)
	{
	  y = *d & ((unsigned) 0xffffffff >> doff);
	  x = *s & (0xffffffff << (32 - soff));
	  *d = x | y;
	}
    }
}
int main()
{
  blit(17, 29, 1000 * 32);
  blit(29, 17, 1000 * 32);
  if (dst[0] != 291 || dst[4] != 1164411171 || dst[10] != 1080066048)
    {
      puts("blit: failed\n");
    } 
  else
    {
      puts("blit: success\n");
    }
  return 0;
}
