#include "common.h"

#define NULL 0

char UnComp[800 + 40];
char CompBuf[800];
static char Buf[] = "/* Replacement routines for standard C routines. */\0#define CONSOLE 0\0#ifndef SUN\0#define stderr CONSOLE\0#define EOF (-1)\0#endif /* SUN */\0\0#include \"buf.c\"\0\0char* outbuf = 0;\0\0int getchar()\0{ static char *bufp = Buf;\0  static int n = Buflen;\0#ifdef TEST\0  if ( n == 0 ) {   /* buffer is empty */\0    n = strtol ( bufp, &bufp, 10 ); /* read char size from 1st string. */\0    }\0#endif TEST\0  return ( --n >= 0 ) ? (unsigned char) *bufp++ : EOF;\0}  \0\0/*void putchar ( c)\0  char c;\0{ \0  fprintf(stderr,\"putchar: c = %x \\n\", c);\0  *outbuf++ = c;\0}\0 */\0#ifndef SUN\0void exit( x )\0  int x;\0{\0  fprintf (stderr, \"exit(0x%x)\\n\", x);\0#ifdef XINU\0  userret();               /* Must link with XINU? */\0#endif /* XINU */\0}\0 \0int putc( dev,  c)   /* putc defined bu XINU. */\0  int dev;\0  char c;\0{\0/* if (dev == CONSOLE)  */\0}\0#endif /* SUN */\0xff\0xff\0xff\0xff\0xff\0xff\0xff\0xff\0";
void compress();
void decompress();
void version();
void cl_block();
int prratio(long int num, long int den);
int Compress(register int, register char **);
extern char *outbuf, *bufp;
extern int buflen;
int main()
{
  int argc;
  char **argv;
  char *v[3];
  char fname[] = "Compress";
  argv = v;
  argc = 1;
  v[0] = fname;
  outbuf = 0;
  bufp = Buf;
  buflen = 800;
  Compress(argc, argv);
  puts("compress: success\n");
  return 0;
}
typedef int code_int;
typedef long int count_int;
typedef unsigned char char_type;
char_type magic_header[] = {"\037\235"};
char *outbuf;
char *bufp;
int buflen;
int n_bits;
int maxbits = 12;
code_int maxcode;
code_int maxmaxcode = 1 << 12;
count_int htab[5003];
unsigned short codetab[5003];
code_int hsize = 5003;
count_int fsize;
code_int free_ent = 0;
int exit_stat = 0;
code_int getcode();
void Usage()
{
  ("Usage: compress [-dfvcV] [-b maxbits] \n");
}
static char rcs_ident[] = "Header: compress.c,v ";
int nomagic = 0;
int zcat_flg = 0;
int quiet = 1;
int block_compress = 0x80;
int clear_flg = 0;
long int ratio = 0;
count_int checkpoint = 10000;
int force = 0;
int (*bgnd_flag) ();
int do_decomp = 0;
static int offset;
long int in_count = 1;
long int bytes_out;
long int out_count = 0;
char *rindex(register char *s, register char c)
{
  char *p;
  for (p = 0; *s; s++)
    if (*s == c)
      p = s;
  return (p);
}

int Compress(argc, argv)
     register int argc;
     char **argv;
{
  char *save_obuf, *save_ibuf, *free_bufp;
  int save_blen;
  char *cp;
  int onintr(), oops();
  do_decomp = 0;
  free_bufp = (char *) CompBuf;
  save_obuf = free_bufp;
  outbuf = save_obuf;
  save_ibuf = bufp;
  save_blen = buflen;

  if ((cp = (char *) rindex(argv[0], '/')) != 0)
    {
      cp++;
    } 
  else
    cp = argv[0];
  for (argc--, argv++; argc > 0; argc--, argv++)
    {
      if (**argv == '-')
	{
	  while (*++(*argv))
	    {
	      switch (**argv)
		{
		case 'V':
		  version();
		  break;
		case 'v':
		  quiet = 0;
		  break;
		case 'd':
		  do_decomp = 1;
		  break;
		case 'n':
		  nomagic = 1;
		  break;
		case 'C':
		  block_compress = 0;
		  break;
		case 'b':
		  if (!(*++(*argv) || (--argc && *++argv)))
		    {
		      ("Missing maxbits\n");
		      Usage();
		      return (1);
		    }
		  goto nextarg;
		case 'c':
		  zcat_flg = 1;
		  break;
		case 'q':
		  quiet = 1;
		  break;
		default:
		  ("Unknown flag: '%c'; ", **argv);
		  Usage();
		  return (1);
		}
	    }
	}
    nextarg:continue;
    }
  if (maxbits < 9)
    maxbits = 9;
  if (maxbits > 12)
    maxbits = 12;
  maxmaxcode = 1 << maxbits;
  hsize = 5003;
  if (fsize < (1 << 12))
    hsize = ((5003 > 5003) ? 5003 : 5003);
  else if (fsize < (1 << 13))
    hsize = ((9001 > 5003) ? 5003 : 9001);
  else if (fsize < (1 << 14))
    hsize = ((18013 > 5003) ? 5003 : 18013);
  else if (fsize < (1 << 15))
    hsize = ((35023 > 5003) ? 5003 : 35023);
  else if (fsize < 47000)
    hsize = ((50021 > 5003) ? 5003 : 50021);
  compress();
  bufp = save_obuf;
  buflen = bytes_out;
  outbuf = (char *) UnComp;
  save_obuf = outbuf;
  if (nomagic == 0)
    {
      if ((((--buflen >= 0) ? (unsigned char) *bufp++ : (-1)) != (magic_header[0] & 0xFF))
          || (((--buflen >= 0) ? (unsigned char) *bufp++ : (-1)) != (magic_header[1] & 0xFF)))
	{
	  ("Data not in compressed format\n");
	  return (1);
	}
      maxbits = ((--buflen >= 0) ? (unsigned char) *bufp++ : (-1));
      block_compress = maxbits & 0x80;
      maxbits &= 0x1f;
      maxmaxcode = 1 << maxbits;
      fsize = 100000;
      if (maxbits > 12)
	{
	  (
	   "stdin: compressed with %d bits, can only handle %d bits\n",
	   maxbits, 12);
	  return (1);
	}
    }
  decompress();
  buflen = save_blen;
  return 0;
}
void cl_hash(register count_int hsize);
void output(register code_int code);
compress()
{
  register long fcode;
  register code_int i = 0;
  register int c;
  register code_int ent;
  register int disp;
  register code_int hsize_reg;
  register int hshift;
  if (nomagic == 0)
    {
      *outbuf++ = magic_header[0];
      *outbuf++ = magic_header[1];
      *outbuf++ = (char) (maxbits | block_compress);
    }
  offset = 0;
  bytes_out = 3;
  out_count = 0;
  clear_flg = 0;
  ratio = 0;
  in_count = 1;
  checkpoint = 10000;
  maxcode = ((1 << (n_bits = 9)) - 1);
  free_ent = ((block_compress) ? 257 : 256);
  ent = ((--buflen >= 0) ? (unsigned char) *bufp++ : (-1));
  hshift = 0;
  for (fcode = (long) hsize; fcode < 65536L; fcode *= 2L)
    hshift++;
  hshift = 8 - hshift;
  hsize_reg = hsize;
  cl_hash((count_int) hsize_reg);
  while ((c = ((--buflen >= 0) ? (unsigned char) *bufp++ : (-1))) != (-1))
    {
      in_count++;
      fcode = (long) (((long) c << maxbits) + ent);
      i = ((c << hshift) ^ ent);
      if (htab[i] == fcode)
	{
	  ent = codetab[i];
	  continue;
	} 
      else if ((long) htab[i] < 0)
	{
	  goto nomatch;
	}
      disp = hsize_reg - i;
      if (i == 0)
	disp = 1;
    probe:
      if ((i -= disp) < 0)
	i += hsize_reg;
      if (htab[i] == fcode)
	{
	  ent = codetab[i];
	  continue;
	}
      if ((long) htab[i] > 0)
	{
	  goto probe;
	}
    nomatch:
      output((code_int) ent);
      out_count++;
      ent = c;
      if (free_ent < maxmaxcode)
	{
	  codetab[i] = free_ent++;
	  htab[i] = fcode;
	} 
      else if ((count_int) in_count >= checkpoint && block_compress)
	{
	  cl_block();
	}
    }
  output((code_int) ent);
  out_count++;
  output((code_int) - 1);
  if (zcat_flg == 0 && !quiet)
    {
      ("Compression: ");
      prratio(in_count - bytes_out, in_count);
    }
  if (bytes_out > in_count)
    exit_stat = 2;
  return 0;
}
static char buf[12];
char_type lmask[9] = {0xff, 0xfe, 0xfc, 0xf8, 0xf0, 0xe0, 0xc0, 0x80, 0x00};
char_type rmask[9] = {0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff};
void output(code)
     register code_int code;
{
  register int r_off = offset, bits = n_bits;
  register char *bp = buf;
  int i;
  if (code >= 0)
    {
      bp += (r_off >> 3);
      r_off &= 7;
      *bp = (*bp & rmask[r_off]) | (code << r_off) & lmask[r_off];
      bp++;
      bits -= (8 - r_off);
      code >>= 8 - r_off;
      if (bits >= 8)
	{
	  *bp++ = code;
	  code >>= 8;
	  bits -= 8;
	}
      if (bits)
	*bp = code;
      offset += n_bits;
      if (offset == (n_bits << 3))
	{
	  bp = buf;
	  bits = n_bits;
	  bytes_out += bits;
	  do
	    {
	      *outbuf++ = *bp;
	      bp++;
	    }
	  while (--bits);
	  offset = 0;
	}
      if (free_ent > maxcode || (clear_flg > 0))
	{
	  if (offset > 0)
	    {
	      for (i = 0; i < n_bits; i++)
		*outbuf++ = buf[i];
	      bytes_out += n_bits;
	    }
	  offset = 0;
	  if (clear_flg)
	    {
	      maxcode = ((1 << (n_bits = 9)) - 1);
	      clear_flg = 0;
	    } 
	  else
	    {
	      n_bits++;
	      if (n_bits == maxbits)
		maxcode = maxmaxcode;
	      else
		maxcode = ((1 << (n_bits)) - 1);
	    }
	}
    } 
  else
    {
      if (offset > 0)
	for (i = 0; i < (offset + 7) / 8; i++)
	  *outbuf++ = buf[i];
      bytes_out += (offset + 7) / 8;
      offset = 0;
    }
}
decompress()
{
  register char_type *stackp;
  register int finchar;
  register code_int code, oldcode, incode;
  maxcode = ((1 << (n_bits = 9)) - 1);
  for (code = 255; code >= 0; code--)
    {
      codetab[code] = 0;
      ((char_type *) (htab))[code] = (char_type) code;
    }
  free_ent = ((block_compress) ? 257 : 256);
  finchar = oldcode = getcode();
  if (oldcode == -1)
    {
      return 0;
    }
  *outbuf++ = (char) finchar;
  stackp = ((char_type *) & ((char_type *) (htab))[1 << 12]);
  while ((code = getcode()) > -1)
    {
      if ((code == 256) && block_compress)
	{
	  for (code = 255; code >= 0; code--)
            codetab[code] = 0;
	  clear_flg = 1;
	  free_ent = 257 - 1;
	  if ((code = getcode()) == -1)
            break;
	}
      incode = code;
      if (code >= free_ent)
	{
	  *stackp++ = finchar;
	  code = oldcode;
	}
      while (code >= 256)
	{
	  *stackp++ = ((char_type *) (htab))[code];
	  code = codetab[code];
	}
      *stackp++ = finchar = ((char_type *) (htab))[code];
      do
	{
	  --stackp;
	  *outbuf++ = *stackp;
	}
      while (stackp > ((char_type *) & ((char_type *) (htab))[1 << 12]));
      if ((code = free_ent) < maxmaxcode)
	{
	  codetab[code] = (unsigned short) oldcode;
	  ((char_type *) (htab))[code] = finchar;
	  free_ent = code + 1;
	}
      oldcode = incode;
    }
  return 0;
}
code_int
getcode()
{
  register code_int code;
  static int offset = 0, size = 0;
  static char_type buf[12];
  register int r_off, bits;
  register char_type *bp = buf;
  if ((clear_flg > 0) || (offset >= size) || (free_ent > maxcode))
    {
      if (free_ent > maxcode)
	{
	  n_bits++;
	  if (n_bits == maxbits)
            maxcode = maxmaxcode;
	  else
            maxcode = ((1 << (n_bits)) - 1);
	}
      if (clear_flg > 0)
	{
	  maxcode = ((1 << (n_bits = 9)) - 1);
	  clear_flg = 0;
	}
      for (size = 0; size < n_bits; size++)
	{
	  buf[size] = ((--buflen >= 0) ? (unsigned char) *bufp++ : (-1));
	  if (buf[size] == (char_type) - 1)
            break;
	}
      if (size <= 0)
	return -1;
      offset = 0;
      size = (size << 3) - (n_bits - 1);
    }
  r_off = offset;
  bits = n_bits;
  bp += (r_off >> 3);
  r_off &= 7;
  code = (*bp++ >> r_off);
  bits -= (8 - r_off);
  r_off = 8 - r_off;
  if (bits >= 8)
    {
      code |= *bp++ << r_off;
      r_off += 8;
      bits -= 8;
    }
  code |= (*bp & rmask[bits]) << r_off;
  offset += n_bits;
  return code;
}
writeerr()
{
  return (1);
  ("ERROR: writerr was called\n");
}
foreground()
{
  if (bgnd_flag)
    {
      return (0);
    } 
  else
    return (1);
}
onintr()
{
  return (1);
}
oops()
{
  if (do_decomp == 1)
    ("uncompress: corrupt input\n");
  return (1);
}
cl_block()
{
  register long int rat;
  checkpoint = in_count + 10000;
  if (in_count > 0x007fffff)
    {
      rat = bytes_out >> 8;
      if (rat == 0)
	{
	  rat = 0x7fffffff;
	} 
      else
	{
	  rat = in_count / rat;
	}
    } 
  else
    {
      rat = (in_count << 8) / bytes_out;
    }
  if (rat > ratio)
    {
      ratio = rat;
    } 
  else
    {
      ratio = 0;
      cl_hash((count_int) hsize);
      free_ent = 257;
      clear_flg = 1;
      output((code_int) 256);
    }
  return 0;
}
void cl_hash(hsize)
     register count_int hsize;
{
  register count_int *htab_p = htab + hsize;
  register long i;
  register long m1 = -1;
  i = hsize - 16;
  do
    {
      *(htab_p - 16) = m1;
      *(htab_p - 15) = m1;
      *(htab_p - 14) = m1;
      *(htab_p - 13) = m1;
      *(htab_p - 12) = m1;
      *(htab_p - 11) = m1;
      *(htab_p - 10) = m1;
      *(htab_p - 9) = m1;
      *(htab_p - 8) = m1;
      *(htab_p - 7) = m1;
      *(htab_p - 6) = m1;
      *(htab_p - 5) = m1;
      *(htab_p - 4) = m1;
      *(htab_p - 3) = m1;
      *(htab_p - 2) = m1;
      *(htab_p - 1) = m1;
      htab_p -= 16;
    }
  while ((i -= 16) >= 0);
  for (i += 16; i > 0; i--)
    *--htab_p = m1;
}
prratio(num, den)
     long int num, den;
{
  register int q;
  if (num > 214748L)
    {
      q = num / (den / 10000L);
    } 
  else
    {
      q = 10000L * num / den;
    }
  if (q < 0)
    {
      ("-");
      q = -q;
    }
  ("%d.%02d%%", q / 100, q % 100);
  return 0;
}
version()
{
  ("%s\n", rcs_ident);
  ("Options: ");
  ("BITS = %d\n", 12);
  return 0;
}
