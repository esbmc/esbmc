#include "common.h"
extern unsigned long int error_corr(register unsigned long int);
extern void fix_bit(register char);
extern void find_syndromes(void);
extern char normalized_locator(void);
extern void validation(void);
extern int comp32(register unsigned long int, register unsigned long int);
extern unsigned long int *sync_find(register unsigned long int *);
extern int addr_corr(register unsigned long int);
extern int num_proc(register int, register unsigned long int);
extern int alpha_proc(register int, register unsigned long int);
extern int msg_proc(register unsigned long int *);
extern int main(void);
unsigned long int hexword, parity;
int error_code;
char s1, s3;
void fix_bit(register char eroot);
static char alpha[] =
{
  0x01, 0x02, 0x04, 0x08, 0x10, 0x05,
  0x0A, 0x14, 0x0D, 0x1A, 0x11, 0x07,
  0x0E, 0x1C, 0x1D, 0x1F, 0x1B, 0x13,
  0x03, 0x06, 0x0C, 0x18, 0x15, 0x0F,
  0x1E, 0x19, 0x17, 0x0B, 0x16, 0x09,
  0x12
};
static char alpha3[] =
{
  0x01, 0x08, 0x0A, 0x1A, 0x0E, 0x1F,
  0x03, 0x18, 0x1E, 0x0B, 0x12, 0x04,
  0x05, 0x0D, 0x07, 0x1D, 0x13, 0x0C,
  0x0F, 0x17, 0x09, 0x02, 0x10, 0x14,
  0x11, 0x1C, 0x1B, 0x06, 0x15, 0x19,
  0x16
};
static char alpha_inv[] =
{
  0x00, 0x00, 0x01, 0x12, 0x02, 0x05,
  0x13, 0x0B, 0x03, 0x1D, 0x06, 0x1B,
  0x14, 0x08, 0x0C, 0x17, 0x04, 0x0A,
  0x1E, 0x11, 0x07, 0x16, 0x1C, 0x1A,
  0x15, 0x19, 0x09, 0x10, 0x0D, 0x0E,
  0x18, 0x0F
};
static char roots[] =
{
  0x00, -128, 0x03, -128, 0x06, -128,
  0x01, -128, -128, 0x07, -128, 0x0D,
  -128, 0x0F, -128, 0x1A, 0x0C, -128,
  0x0B, -128, 0x02, -128, 0x08, -128,
  -128, 0x09, -128, 0x11, -128, 0x04,
  -128, 0x15
};
unsigned long int err_corr(register unsigned long int datain)
{
  char locator_constant, root2;
  signed char root1;
  hexword = datain;
  find_syndromes();
  if (s1 == 0 && s3 == 0)
    error_code = 0;
  else
    {
      locator_constant = normalized_locator();
      if (locator_constant == 0)
	{
	  root1 = alpha_inv[s1];
	  fix_bit(root1);
	  error_code = 1;
	} else
	  {
	    root1 = roots[locator_constant];
	    if (root1 < 0)
	      error_code = 4;
	    else
	      {
		root1 = alpha_inv[s1] + root1;
		while (root1 > 30)
		  root1 -= 31;
		fix_bit(root1);
		root2 = s1 ^ alpha[root1];
		root2 = alpha_inv[root2];
		fix_bit(root2);
		if (parity)
		  error_code = 3;
		else
		  error_code = 2;
	      }
	  }
    }
  return (hexword);
}
void fix_bit(register char eroot)
{
  eroot++;
  hexword = hexword ^ (0x00000001 << eroot);
}
void find_syndromes()
{
  unsigned long int mask = 0x80000000;
  int i;
  s1 = 0;
  s3 = 0;
  parity = hexword & 0x00000001;
  i = 30;
  while (i >= 0)
    {
      if (hexword & mask)
	{
	  s1 ^= alpha[i];
	  s3 ^= alpha3[i];
	  parity++;
	}
      mask >>= 1;
      i--;
    }
  parity &= 0x00000001;
}
char normalized_locator()
{
  char power_s1;
  signed char tau;
  power_s1 = alpha_inv[s1];
  tau = alpha_inv[s3] - power_s1 - power_s1 - power_s1;
  while (tau < 0)
    tau += 31;
  return (alpha[tau] ^ 1);
}
extern int error_code;
unsigned char msg[256];
int alpha_count = 0;
int func;
static unsigned char err_tab[] =
{
  0x0, 0x1, 0x1, 0x2, 0x1, 0x2, 0x2, 0x3, 0x1, 0x2, 0x2, 0x3, 0x2, 0x3, 0x3, 0x4,
  0x1, 0x2, 0x2, 0x3, 0x2, 0x3, 0x3, 0x4, 0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5,
  0x1, 0x2, 0x2, 0x3, 0x2, 0x3, 0x3, 0x4, 0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5,
  0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5, 0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6,
  0x1, 0x2, 0x2, 0x3, 0x2, 0x3, 0x3, 0x4, 0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5,
  0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5, 0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6,
  0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5, 0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6,
  0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6, 0x4, 0x5, 0x5, 0x6, 0x5, 0x6, 0x6, 0x7,
  0x1, 0x2, 0x2, 0x3, 0x2, 0x3, 0x3, 0x4, 0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5,
  0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5, 0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6,
  0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5, 0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6,
  0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6, 0x4, 0x5, 0x5, 0x6, 0x5, 0x6, 0x6, 0x7,
  0x2, 0x3, 0x3, 0x4, 0x3, 0x4, 0x4, 0x5, 0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6,
  0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6, 0x4, 0x5, 0x5, 0x6, 0x5, 0x6, 0x6, 0x7,
  0x3, 0x4, 0x4, 0x5, 0x4, 0x5, 0x5, 0x6, 0x4, 0x5, 0x5, 0x6, 0x5, 0x6, 0x6, 0x7,
  0x4, 0x5, 0x5, 0x6, 0x5, 0x6, 0x6, 0x7, 0x5, 0x6, 0x6, 0x7, 0x6, 0x7, 0x7, 0x8
};
int comp32(register unsigned long int sample, register unsigned long int reference)
{
  int i, num_err;
  sample = sample ^ reference;
  num_err = 0;
  i = 4;
  while (i > 0)
    {
      num_err = num_err +
	err_tab[(unsigned char) sample];
      sample = sample >> 8;
      i--;
    }
  return (num_err);
}
unsigned long int *sync_find(register unsigned long int *dptr)
{
  int num_err;
  int sync_window = 18;
  int bit_cnt;
  unsigned long int sample1, sample2;
  sample1 = *dptr++;
  while (sync_window > 0)
    {
      sample2 = *dptr++;
      bit_cnt = 0;
      while (bit_cnt < 32)
	{
	  num_err = comp32(sample1, 0x3e4ba81b);
	  if (num_err <= 2)
            return (dptr);
	  num_err = comp32(sample1, 0xaaaaaaaa);
	  if ((num_err <= 2) || (num_err >= 30))
	    {
	      sample1 = sample2 >> bit_cnt;
	      sync_window = 19;
	      break;
	    } 
	  else
	    {
	      sample1 = (sample1 << 1) |
		(sample2 >> 31);
	      sample2 = sample2 << 1;
	    }
	  bit_cnt++;
	}
      sync_window--;
    }
  return (0);
}
int addr_corr(register unsigned long int data)
{
  int num_err;
  unsigned long int addr;
  unsigned long int capcode = 0x2a74e;
  addr = data >> 13;
  num_err = comp32(addr, capcode);
  if (num_err > 2)
    {
      capcode = 0x1d25a;
      num_err = comp32(addr, capcode);
    }
  if (num_err <= 2)
    {
      data = err_corr(data);
      if (error_code < 3)
	{
	  if (error_code != 0)
	    {
	      addr = data >> 13;
	      num_err = comp32(addr, capcode);
	    }
	  if (num_err == 0)
	    {
	      func = (unsigned char) ((data >> 11) &
				      0x3);
	      return (1);
	    }
	}
    }
  return (0);
}
int num_proc(register int i, register unsigned long int codeword)
{
  int count = 5;
  int shift = 0;
  unsigned char digit;
  while (count > 0)
    {
      digit = (unsigned char) ((codeword >> shift) &
                               0xf);
      if (error_code < 3)
	{
	  if (digit < 0xa)
            msg[i] = digit + 0x30;
	  else
            msg[i] = digit;
	} 
      else
	msg[i] = 0x80;
      shift = shift + 4;
      i++;
      count--;
    }
  return (i);
}
int alpha_proc(register int i, register unsigned long int codeword)
{
  int num_bits_left;
  unsigned char rem_bits;
  int char_count = 3;
  int char_shift = 0;
  if (alpha_count > 0)
    {
      num_bits_left = 7 - alpha_count;
      rem_bits = (unsigned char) (codeword
                                  << num_bits_left);
      if (error_code < 3)
	msg[i] = (msg[i] | rem_bits) &
          0x7f;
      else
	msg[i] = 0x80;
      i++;
      char_shift = alpha_count;
    }
  while (char_count > 0)
    {
      if (error_code < 3)
	msg[i] = (unsigned char) ((codeword >> char_shift) &
				  0x7f);
      else
	msg[i] = 0x80;
      char_shift = char_shift + 7;
      i++;
      char_count--;
    }
  i--;
  if (alpha_count == 6)
    alpha_count = 0;
  else
    alpha_count++;
  return (i);
}
int msg_proc(register unsigned long int *dptr)
{
  int i = 0;
  int word_count = 0;
  int num_bad_sync = 0;
  int num_err;
  int addr_search_mode = 1;
  unsigned long int addr;
  unsigned long int data;
  while (1)
    {
      data = *dptr++;
      if (word_count == 16)
	{
	  num_err = comp32(data, 0x3e4ba81b);
	  if (num_err <= 2)
            num_bad_sync = 0;
	  else
	    {
	      if (num_bad_sync == 1)
		return (i);
	      else
		num_bad_sync++;
	    }
	  word_count = 0;
	} 
      else
	{
	  if (addr_search_mode)
	    {
	      if ((word_count == 6) ||
		  (word_count == 7))
		{
		  if (addr_corr(data))
		    {
		      addr_search_mode = 0;
		    }
		}
	    } 
	  else
	    {
	      data = err_corr(data);
	      if ((word_count == 6) ||
		  (word_count == 7))
		{
		  num_err = 1;
		  if (error_code < 3)
		    {
		      addr = data >> 13;
		      num_err = comp32(addr, 0x2a74e);
		      if (num_err)
			num_err = comp32(addr, 0x1d25a);
		    }
		  if (num_err == 0)
		    return (i);
		}
	      data = (data >> 11) & 0xfffff;
	      switch (func)
		{
		case 0:
		  i = num_proc(i, data);
		  break;
		case 3:
		  i = alpha_proc(i, data);
		  break;
		default:
		  return (0);
		}
	    }
	  word_count++;
	}
    }
  return 0;
}
unsigned long alpha_data[] = {
  0xaa8a2aaa,
  0x55551545,
  0x11555555,
  0xaaa22aaa,
  0x2a8aaaaa,
  0x55415555,
  0xafaaaaaa,
  0xf5555555,
  0xaa2aaaaa,
  0x5555555f,
  0x50555555,
  0x51555545,
  0xaaaa0aaa,
  0x2aaaaaa8,
  0xaae2aaaa,
  0x5555053e,
  0x4ba90b00,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x3a4b5832,
  0xc396212e,
  0x99072006,
  0xe688b86e,
  0x8efbf05e,
  0x90e5a59e,
  0xf120f237,
  0xe3dd935e,
  0xc3b146cd,
  0x3e0ba81b,
  0x975f3e6d,
  0xd2ce5ebe,
  0xfb2281a8,
  0x883aeea9,
  0xfb61d407,
  0xea4191c5,
  0xe9cfabbb,
  0x97221e4b,
  0xa6fc78b4,
  0x8c9972c8,
  0x9bb1a0f9,
  0xdc3847cf,
  0xe8c4b1e1,
  0xc3cb0f25,
  0x97a63952,
  0x8a0e4d29,
  0x3e4ba81b,
  0x8e1b30ca,
  0xbbb0ecfa,
  0xb2ede9bf,
  0xe3d3c8d9,
  0xcbb503f0,
  0x879f3efb,
  0x8e5cfaee,
  0x54e9d8d5
};
unsigned long numeric_data[] = {
  0xaa8a2aaa,
  0x55551545,
  0x11555555,
  0xaaa22aaa,
  0x2a8aaaaa,
  0x55415555,
  0xafaaaaaa,
  0xf5555555,
  0xaa2aaaaa,
  0x5555555f,
  0x50555555,
  0x51555545,
  0xaaaa0aaa,
  0x2aaaaaa8,
  0xaae2aaaa,
  0x5555053e,
  0x4b801b00,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x380b45a7,
  0xbeb804bb,
  0xc4689285,
  0xdfe2a81b,
  0xe66474a7,
  0xe66667d1,
  0xe66627d1,
  0xe66667d1,
  0xe66667d1,
  0x3e4b881b,
  0xe66667d1,
  0xe66667d1,
  0xe66667d1,
  0xe66667d1,
  0xe66667d1,
  0xe66667d1,
  0x54e8d9d5
};
int main()
{
  unsigned long int *dptr;
  int msg_length;
  int i;
  int j;
  for (j = 0; j < 2; j++)
    {
      alpha_count = 0;
      if (!j)
	{
	  dptr = numeric_data;
	} 
      else
	{
	  dptr = alpha_data;
	}
      if (sync_find(dptr) != 0)
	{
	  msg_length = msg_proc(sync_find(dptr));
	  if (func == 0)
	    {
	    } 
	  else if (func == 3)
	    {
	    } 
	  else
	    {
	      puts("pocsag: failed\n");
	      return 0;
	    }
	} 
      else
	{
	  puts("pocsag: failed\n");
	  return 0;
	}
    }
  if (strncmp((char *) msg, "Dear fellow ACP benchmarker", 27) == 0 &&
      msg_length == 88)
    {
      puts("pocsag: success\n");
    } 
  else
    {
      puts("pocsag: failed\n");
    }
  return 0;
}
