//#include "common.h"
#include <stdio.h>

typedef unsigned char uchar;

unsigned char aa[] = "asdffeagewaHAFEFaeDsFEawFdsFaefaeerdjgpim23";

// unsigned char aa[] = "abcdefghijklmnopqrstuvwxyz0123456789abcdefg";

unsigned short icrc1(unsigned short crc, unsigned char onech)
{
  int i;
  unsigned short ans = (crc ^ onech << 8);
  for (i = 0; i < 8; i++)
    {
      if (ans & 0x8000)
	ans = (ans <<= 1) ^ 4129;
      else
	ans <<= 1;
    }
  return ans;
}

unsigned short icrc(unsigned short crc, unsigned char *lin, unsigned int len, short jinit, int jrev)
{
  static unsigned short icrctb[256], init = 0;
  static uchar rchr[256];
  unsigned short tmp1, tmp2, j, cword = crc;
  static uchar it[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  
  if (!init)
  {
      init = 1;
      for (j = 0; j <= 255; j++)
      {
    	  icrctb[j] = icrc1(j << 8, (uchar) 0);
    	  rchr[j] = (uchar) (it[j & 0xF] << 4 | it[j >> 4]);
      }
  }
  
  if (jinit >= 0)
  {
    cword = ((uchar) jinit) | (((uchar) jinit) << 8);
  }
  else 
	  {
	  	if (jrev < 0)
	  	{
	  		cword = rchr[((uchar) ((cword) >> 8))] | rchr[((uchar) ((cword) & 0xFF))] << 8;
	  	}
	  }
    
  
  for (j = 1; j <= len; j++)
  {
      if (jrev < 0)
      {
    	  tmp1 = rchr[lin[j]] ^ ((uchar) ((cword) >> 8));
      } 
      else
      {
    	  tmp1 = lin[j] ^ ((uchar) ((cword) >> 8));
      }
      cword = icrctb[tmp1] ^ ((uchar) ((cword) & 0xFF)) << 8;
  }
  
  if (jrev >= 0)
  {
      tmp2 = cword;
  } 
  else
  {
      tmp2 = rchr[((uchar) ((cword) >> 8))] | rchr[((uchar) ((cword) & 0xFF))] << 8;
  }
  
  return (tmp2);
}

//int argc, char *argv[]
int main()
{
  unsigned short i1, i2;
  int n;

  i1 = icrc(0, aa, 40, (short) 0, 1);
  i2 = icrc(i1, aa, 42, (short) -1, 1);
  

	printf("i1: %d\t i2:%d\n", i1, i2);
  if (i2 != 268)
  {
      puts("crc: fail\n");
  } 
  else
    {
      puts("crc: success\n");
    }
  
  return 0;
  
}
