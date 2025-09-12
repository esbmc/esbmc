#ifndef ASMINLINE
#line 2 "../src/kernel/none/addll.h"
/* Copyright (C) 2003  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

/* This file originally adapted from gmp-3.1.1 (from T. Granlund), files
 * longlong.h and gmp-impl.h

  Copyright (C) 2000 Free Software Foundation, Inc. */

#undef LOCAL_OVERFLOW
#define LOCAL_OVERFLOW
extern ulong overflow;

#if !defined(INLINE)
extern long addll(ulong x, ulong y);
extern long addllx(ulong x, ulong y);
extern long subll(ulong x, ulong y);
extern long subllx(ulong x, ulong y);
#else

#if defined(__GNUC__) && !defined(DISABLE_INLINE)
#undef LOCAL_OVERFLOW
#define LOCAL_OVERFLOW ulong overflow

#define addll(a, b)                                             \
__extension__ ({                                                \
   ulong __arg1 = (a), __arg2 = (b), __value = __arg1 + __arg2; \
   overflow = (__value < __arg1);                               \
   __value;                                                     \
})

#define addllx(a, b)                                          \
__extension__ ({                                              \
   ulong __arg1 = (a), __arg2 = (b), __value, __tmp = __arg1 + overflow;\
   overflow = (__tmp < __arg1);                               \
   __value = __tmp + __arg2;                                  \
   overflow |= (__value < __tmp);                             \
   __value;                                                   \
})

#define subll(a, b)                                           \
__extension__ ({                                              \
   ulong __arg1 = (a), __arg2 = (b);                          \
   overflow = (__arg2 > __arg1);                              \
   __arg1 - __arg2;                                           \
})

#define subllx(a, b)                                  \
__extension__ ({                                      \
   ulong __arg1 = (a), __arg2 = (b), __value, __tmp = __arg1 - overflow;\
   overflow = (__arg1 < overflow);                    \
   __value = __tmp - __arg2;                          \
   overflow |= (__arg2 > __tmp);                      \
   __value;                                           \
})

#else /* __GNUC__ */

INLINE long
addll(ulong x, ulong y)
{
  const ulong z = x+y;
  overflow=(z<x);
  return (long) z;
}

INLINE long
addllx(ulong x, ulong y)
{
  const ulong z = x+y+overflow;
  overflow = (z<x || (z==x && overflow));
  return (long) z;
}

INLINE long
subll(ulong x, ulong y)
{
  const ulong z = x-y;
  overflow = (z>x);
  return (long) z;
}

INLINE long
subllx(ulong x, ulong y)
{
  const ulong z = x-y-overflow;
  overflow = (z>x || (z==x && overflow));
  return (long) z;
}

#endif /* __GNUC__ */

#endif
#line 2 "../src/kernel/none/mulll.h"
/* Copyright (C) 2000  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

#undef  LOCAL_HIREMAINDER
#define LOCAL_HIREMAINDER
extern ulong hiremainder;

/* Version Peter Montgomery */
/*
 *      Assume (for presentation) that BITS_IN_LONG = 32.
 *      Then 0 <= xhi, xlo, yhi, ylo <= 2^16 - 1.  Hence
 *
 * -2^31 + 2^16 <= (xhi-2^15)*(ylo-2^15) + (xlo-2^15)*(yhi-2^15) <= 2^31.
 *
 *      If xhi*ylo + xlo*yhi = 2^32*overflow + xymid, then
 *
 * -2^32 + 2^16 <= 2^32*overflow + xymid - 2^15*(xhi + ylo + xlo + yhi) <= 0.
 *
 * 2^16*overflow <= (xhi+xlo+yhi+ylo)/2 - xymid/2^16 <= 2^16*overflow + 2^16-1
 *
 *       This inequality was derived using exact (rational) arithmetic;
 *       it remains valid when we truncate the two middle terms.
 */

#if !defined(INLINE)
extern long mulll(ulong x, ulong y);
extern long addmul(ulong x, ulong y);
#else

#if defined(__GNUC__) && !defined(DISABLE_INLINE)
#undef LOCAL_HIREMAINDER
#define LOCAL_HIREMAINDER ulong hiremainder

#define mulll(x, y) \
__extension__ ({ \
  const ulong __x = (x), __y = (y);\
  const ulong __xlo = LOWWORD(__x), __xhi = HIGHWORD(__x); \
  const ulong __ylo = LOWWORD(__y), __yhi = HIGHWORD(__y); \
  ulong __xylo,__xymid,__xyhi,__xymidhi,__xymidlo; \
  ulong __xhl,__yhl; \
 \
  __xylo = __xlo*__ylo; __xyhi = __xhi*__yhi; \
  __xhl = __xhi+__xlo; __yhl = __yhi+__ylo; \
  __xymid = __xhl*__yhl - (__xyhi+__xylo); \
 \
  __xymidhi = HIGHWORD(__xymid); \
  __xymidlo = __xymid << BITS_IN_HALFULONG; \
 \
  __xylo += __xymidlo; \
  hiremainder = __xyhi + __xymidhi + (__xylo < __xymidlo) \
     + ((((__xhl + __yhl) >> 1) - __xymidhi) & HIGHMASK); \
 \
  __xylo; \
})

#define addmul(x, y) \
__extension__ ({                                           \
  const ulong __x = (x), __y = (y);\
  const ulong __xlo = LOWWORD(__x), __xhi = HIGHWORD(__x); \
  const ulong __ylo = LOWWORD(__y), __yhi = HIGHWORD(__y); \
  ulong __xylo,__xymid,__xyhi,__xymidhi,__xymidlo; \
  ulong __xhl,__yhl; \
 \
  __xylo = __xlo*__ylo; __xyhi = __xhi*__yhi; \
  __xhl = __xhi+__xlo; __yhl = __yhi+__ylo; \
  __xymid = __xhl*__yhl - (__xyhi+__xylo); \
 \
  __xylo += hiremainder; __xyhi += (__xylo < hiremainder); \
 \
  __xymidhi = HIGHWORD(__xymid); \
  __xymidlo = __xymid << BITS_IN_HALFULONG; \
 \
  __xylo += __xymidlo; \
  hiremainder = __xyhi + __xymidhi + (__xylo < __xymidlo) \
     + ((((__xhl + __yhl) >> 1) - __xymidhi) & HIGHMASK); \
 \
  __xylo; \
})

#else

INLINE long
mulll(ulong x, ulong y)
{
  const ulong xlo = LOWWORD(x), xhi = HIGHWORD(x);
  const ulong ylo = LOWWORD(y), yhi = HIGHWORD(y);
  ulong xylo,xymid,xyhi,xymidhi,xymidlo;
  ulong xhl,yhl;

  xylo = xlo*ylo; xyhi = xhi*yhi;
  xhl = xhi+xlo; yhl = yhi+ylo;
  xymid = xhl*yhl - (xyhi+xylo);

  xymidhi = HIGHWORD(xymid);
  xymidlo = xymid << BITS_IN_HALFULONG;

  xylo += xymidlo;
  hiremainder = xyhi + xymidhi + (xylo < xymidlo)
     + ((((xhl + yhl) >> 1) - xymidhi) & HIGHMASK);

  return xylo;
}

INLINE long
addmul(ulong x, ulong y)
{
  const ulong xlo = LOWWORD(x), xhi = HIGHWORD(x);
  const ulong ylo = LOWWORD(y), yhi = HIGHWORD(y);
  ulong xylo,xymid,xyhi,xymidhi,xymidlo;
  ulong xhl,yhl;

  xylo = xlo*ylo; xyhi = xhi*yhi;
  xhl = xhi+xlo; yhl = yhi+ylo;
  xymid = xhl*yhl - (xyhi+xylo);

  xylo += hiremainder; xyhi += (xylo < hiremainder);

  xymidhi = HIGHWORD(xymid);
  xymidlo = xymid << BITS_IN_HALFULONG;

  xylo += xymidlo;
  hiremainder = xyhi + xymidhi + (xylo < xymidlo)
     + ((((xhl + yhl) >> 1) - xymidhi) & HIGHMASK);

  return xylo;
}
#endif

#endif
#line 2 "../src/kernel/none/bfffo.h"
/* Copyright (C) 2000  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

#if !defined(INLINE)
extern int  bfffo(ulong x);
#else

#if defined(__GNUC__) && !defined(DISABLE_INLINE)

#ifdef LONG_IS_64BIT
#  define bfffo(x) \
__extension__ ({ \
  static int __bfffo_tabshi[16]={4,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0};\
  int __value = BITS_IN_LONG - 4; \
  ulong __arg1=(x); \
  if (__arg1 & ~0xffffffffUL) {__value -= 32; __arg1 >>= 32;}\
  if (__arg1 & ~0xffffUL) {__value -= 16; __arg1 >>= 16;} \
  if (__arg1 & ~0x00ffUL) {__value -= 8; __arg1 >>= 8;} \
  if (__arg1 & ~0x000fUL) {__value -= 4; __arg1 >>= 4;} \
  __value + __bfffo_tabshi[__arg1]; \
})
#else
#  define bfffo(x) \
__extension__ ({ \
  static int __bfffo_tabshi[16]={4,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0};\
  int __value = BITS_IN_LONG - 4; \
  ulong __arg1=(x); \
  if (__arg1 & ~0xffffUL) {__value -= 16; __arg1 >>= 16;} \
  if (__arg1 & ~0x00ffUL) {__value -= 8; __arg1 >>= 8;} \
  if (__arg1 & ~0x000fUL) {__value -= 4; __arg1 >>= 4;} \
  __value + __bfffo_tabshi[__arg1]; \
})
#endif

#else

INLINE int
bfffo(ulong x)
{
  static int tabshi[16]={4,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0};
  int value = BITS_IN_LONG - 4;
  ulong arg1=x;
#ifdef LONG_IS_64BIT
  if (arg1 & ~0xffffffffUL) {value -= 32; arg1 >>= 32;}
#endif
  if (arg1 & ~0xffffUL) {value -= 16; arg1 >>= 16;}
  if (arg1 & ~0x00ffUL) {value -= 8; arg1 >>= 8;}
  if (arg1 & ~0x000fUL) {value -= 4; arg1 >>= 4;}
  return value + tabshi[arg1];
}
#endif

#endif
#line 2 "../src/kernel/none/divll.h"
/* Copyright (C) 2003  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

/* This file originally adapted from gmp-3.1.1 (from T. Granlund), files
 * longlong.h and gmp-impl.h

  Copyright (C) 2000 Free Software Foundation, Inc. */

#undef  LOCAL_HIREMAINDER
#define LOCAL_HIREMAINDER
extern ulong hiremainder;

#if !defined(INLINE)
extern long divll(ulong x, ulong y);
#else

#define __GLUE(hi, lo) (((hi) << BITS_IN_HALFULONG) | (lo))
#define __SPLIT(a, b, c) b = HIGHWORD(a); c = LOWWORD(a)
#define __LDIV(a, b, q, r) q = a / b; r = a - q*b
extern ulong hiremainder;

/* divide (hiremainder * 2^BITS_IN_LONG + n0) by d; assume hiremainder < d.
 * Return quotient, set hiremainder to remainder */

#if defined(__GNUC__) && !defined(DISABLE_INLINE)
#undef LOCAL_HIREMAINDER
#define LOCAL_HIREMAINDER ulong hiremainder

#define divll(n0, d)                                                    \
__extension__ ({                                                        \
  ulong __d1, __d0, __q1, __q0, __r1, __r0, __m, __n1, __n0;            \
  ulong __k, __d;                                                       \
                                                                        \
  __n1 = hiremainder; __n0 = n0; __d = d;                               \
  if (__n1 == 0)                                                        \
  { /* Only one division needed */                                      \
    __LDIV(__n0, __d, __q1, hiremainder);                               \
  }                                                                     \
  else if (__d < LOWMASK)                                               \
  { /* Two half-word divisions  */                                      \
    __n1 = __GLUE(__n1, HIGHWORD(__n0));                                \
    __LDIV(__n1, __d, __q1, __r1);                                      \
    __n1 = __GLUE(__r1,  LOWWORD(__n0));                                \
    __LDIV(__n1, __d, __q0, hiremainder);                               \
    __q1 = __GLUE(__q1, __q0);                                          \
  }                                                                     \
  else                                                                  \
  { /* General case */                                                  \
    if (__d & HIGHBIT)                                                  \
    {                                                                   \
      __k = 0; __SPLIT(__d, __d1, __d0);                                \
    }                                                                   \
    else                                                                \
    {                                                                   \
      __k = bfffo(__d);                                                 \
      __n1 = (__n1 << __k) | (__n0 >> (BITS_IN_LONG - __k));            \
      __n0 <<= __k;                                                     \
      __d = __d << __k; __SPLIT(__d, __d1, __d0);                       \
    }                                                                   \
    __LDIV(__n1, __d1, __q1, __r1);                                     \
    __m =  __q1 * __d0;                                                 \
    __r1 = __GLUE(__r1, HIGHWORD(__n0));                                  \
    if (__r1 < __m)                                                        \
    {                                                                        \
      __q1--, __r1 += __d;                                                \
      if (__r1 >= __d) /* we didn't get carry when adding to __r1 */    \
        if (__r1 < __m)        __q1--, __r1 += __d;                                \
    }                                                                        \
    __r1 -= __m;                                                        \
    __LDIV(__r1, __d1, __q0, __r0);                                     \
    __m =  __q0 * __d0;                                                  \
    __r0 = __GLUE(__r0, LOWWORD(__n0));                                   \
    if (__r0 < __m)                                                        \
    {                                                                        \
      __q0--, __r0 += __d;                                                \
      if (__r0 >= __d)                                                        \
        if (__r0 < __m)        __q0--, __r0 += __d;                                \
    }                                                                        \
    hiremainder = (__r0 - __m) >> __k;                                        \
    __q1 = __GLUE(__q1, __q0);                                                 \
  }                                                                           \
  __q1;                                                                        \
})

#else /* __GNUC__ */

INLINE long
divll(ulong n0, ulong d)
{
  ulong __d1, __d0, __q1, __q0, __r1, __r0, __m, __n1, __n0;
  ulong __k, __d;

  __n1 = hiremainder; __n0 = n0; __d = d;

  if (__n1 == 0)
  { /* Only one division needed */
    __LDIV(__n0, __d, __q1, hiremainder);
  }
  else if (__d < LOWMASK)
  { /* Two half-word divisions  */
    __n1 = __GLUE(__n1, HIGHWORD(__n0));
    __LDIV(__n1, __d, __q1, __r1);
    __n1 = __GLUE(__r1,  LOWWORD(__n0));
    __LDIV(__n1, __d, __q0, hiremainder);
    __q1 = __GLUE(__q1, __q0);
  }
  else
  { /* General case */
    if (__d & HIGHBIT)
    {
      __k = 0; __SPLIT(__d, __d1, __d0);
    }
    else
    {
      __k = bfffo(__d);
      __n1 = (__n1 << __k) | (__n0 >> (BITS_IN_LONG - __k));
      __n0 = __n0 << __k;
      __d = __d << __k; __SPLIT(__d, __d1, __d0);
    }
    __LDIV(__n1, __d1, __q1, __r1);
    __m =  __q1 * __d0;
    __r1 = __GLUE(__r1, HIGHWORD(__n0));
    if (__r1 < __m)
      {
        __q1--, __r1 += __d;
        if (__r1 >= __d) /* we didn't get carry when adding to __r1 */
          if (__r1 < __m) __q1--, __r1 += __d;
      }
    __r1 -= __m;
    __LDIV(__r1, __d1, __q0, __r0);
    __m =  __q0 * __d0;
    __r0 = __GLUE(__r0, LOWWORD(__n0));
    if (__r0 < __m)
      {
        __q0--, __r0 += __d;
        if (__r0 >= __d)
          if (__r0 < __m) __q0--, __r0 += __d;
      }
    hiremainder = (__r0 - __m) >> __k;
    __q1 = __GLUE(__q1, __q0);
  }
  return __q1;
}

#endif /* __GNUC__ */

#endif
#endif
#line 2 "../src/kernel/x86-64/asm0.h"
/* Copyright (C) 2004  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

/*
ASM addll mulll bfffo divll
*/
/* Written by Bill Allombert from the ix86 version by Bruno Haible. Basically
 * change insl to insq*/
#ifdef ASMINLINE
#define LOCAL_HIREMAINDER  ulong hiremainder
#define LOCAL_OVERFLOW     ulong overflow

#define addll(a,b) \
__extension__ ({ ulong __value, __arg1 = (a), __arg2 = (b); \
   __asm__ ("addq %3,%0 ; adcq %1,%1" \
        : "=r" (__value), "=r" (overflow) \
        : "0" (__arg1), "g" (__arg2), "1" ((ulong)0) \
        : "cc"); \
  __value; \
})

#define addllx(a,b) \
__extension__ ({ ulong __value, __arg1 = (a), __arg2 = (b), __temp; \
   __asm__ ("subq %5,%2 ; adcq %4,%0 ; adcq %1,%1" \
        : "=r" (__value), "=&r" (overflow), "=&r" (__temp) \
        : "0" (__arg1), "g" (__arg2), "g" (overflow), "1" ((ulong)0), "2" ((ulong)0) \
        : "cc"); \
  __value; \
})

#define addllx8(a,b,c,overflow) \
do { long *__arg1 = a, *__arg2 = b, *__out = c; \
     ulong __temp; \
   __asm__ ("subq %5, %0 \n\t" \
            "movq    (%2), %0 ; adcq    (%3),%0; movq %0,    (%4) \n\t" \
            "movq  -8(%2), %0 ; adcq  -8(%3),%0; movq %0,  -8(%4) \n\t" \
            "movq -16(%2), %0 ; adcq -16(%3),%0; movq %0, -16(%4) \n\t" \
            "movq -24(%2), %0 ; adcq -24(%3),%0; movq %0, -24(%4) \n\t" \
            "movq -32(%2), %0 ; adcq -32(%3),%0; movq %0, -32(%4) \n\t" \
            "movq -40(%2), %0 ; adcq -40(%3),%0; movq %0, -40(%4) \n\t" \
            "movq -48(%2), %0 ; adcq -48(%3),%0; movq %0, -48(%4) \n\t" \
            "movq -56(%2), %0 ; adcq -56(%3),%0; movq %0, -56(%4) \n\t" \
            "adcq  %1, %1" \
        : "=&r" (__temp), "=&r" (overflow) \
        : "r" (__arg1), "r" (__arg2), "r" (__out), "g" (overflow), "0" ((ulong)0), "1" ((ulong)0) \
        : "cc"); \
} while(0)

#define subll(a,b) \
__extension__ ({ ulong __value, __arg1 = (a), __arg2 = (b); \
   __asm__ ("subq %3,%0 ; adcq %1,%1" \
        : "=r" (__value), "=r" (overflow) \
        : "0" (__arg1), "g" (__arg2), "1" ((ulong)0) \
        : "cc"); \
  __value; \
})

#define subllx(a,b) \
__extension__ ({ ulong __value, __arg1 = (a), __arg2 = (b), __temp; \
   __asm__ ("subq %5,%2 ; sbbq %4,%0 ; adcq %1,%1" \
        : "=r" (__value), "=&r" (overflow), "=&r" (__temp) \
        : "0" (__arg1), "g" (__arg2), "g" (overflow), "1" ((ulong)0), "2" ((ulong)0) \
        : "cc"); \
  __value; \
})

#define subllx8(a,b,c,overflow) \
do { long *__arg1 = a, *__arg2 = b, *__out = c; \
     ulong __temp; \
   __asm__ ("subq %5, %0 \n\t" \
            "movq    (%2), %0 ; sbbq    (%3),%0; movq %0,    (%4) \n\t" \
            "movq  -8(%2), %0 ; sbbq  -8(%3),%0; movq %0,  -8(%4) \n\t" \
            "movq -16(%2), %0 ; sbbq -16(%3),%0; movq %0, -16(%4) \n\t" \
            "movq -24(%2), %0 ; sbbq -24(%3),%0; movq %0, -24(%4) \n\t" \
            "movq -32(%2), %0 ; sbbq -32(%3),%0; movq %0, -32(%4) \n\t" \
            "movq -40(%2), %0 ; sbbq -40(%3),%0; movq %0, -40(%4) \n\t" \
            "movq -48(%2), %0 ; sbbq -48(%3),%0; movq %0, -48(%4) \n\t" \
            "movq -56(%2), %0 ; sbbq -56(%3),%0; movq %0, -56(%4) \n\t" \
            "adcq  %1, %1" \
        : "=&r" (__temp), "=&r" (overflow) \
        : "r" (__arg1), "r" (__arg2), "r" (__out), "g" (overflow), "0" ((ulong)0), "1" ((ulong)0) \
        : "cc"); \
} while(0)

#define mulll(a,b) \
__extension__ ({ ulong __valuelo, __arg1 = (a), __arg2 = (b); \
   __asm__ ("mulq %3" \
        : "=a" /* %eax */ (__valuelo), "=d" /* %edx */ (hiremainder) \
        : "0" (__arg1), "rm" (__arg2)); \
   __valuelo; \
})

#define addmul(a,b) \
__extension__ ({ ulong __valuelo, __arg1 = (a), __arg2 = (b), __temp; \
   __asm__ ("mulq %4 ; addq %5,%0 ; adcq %6,%1" \
        : "=a" /* %eax */ (__valuelo), "=&d" /* %edx */ (hiremainder), "=r" (__temp) \
        : "0" (__arg1), "rm" (__arg2), "g" (hiremainder), "2" ((ulong)0)); \
   __valuelo; \
})

#define divll(a,b) \
__extension__ ({ ulong __value, __arg1 = (a), __arg2 = (b); \
   __asm__ ("divq %4" \
        : "=a" /* %eax */ (__value), "=d" /* %edx */ (hiremainder) \
        : "0" /* %eax */ (__arg1), "1" /* %edx */ (hiremainder), "mr" (__arg2)); \
   __value; \
})

#define bfffo(x) \
__extension__ ({ ulong __arg = (x); \
   long leading_one_position; \
  __asm__ ("bsrq %1,%0" : "=r" (leading_one_position) : "rm" (__arg)); \
  63 - leading_one_position; \
})
#endif
#line 2 "../src/kernel/none/divll_pre.h"
/* Copyright (C) 2014  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

#undef  LOCAL_HIREMAINDER
extern ulong hiremainder;
#if defined(INLINE) && defined(__GNUC__) && !defined(DISABLE_INLINE)
#define LOCAL_HIREMAINDER ulong hiremainder
#else
#define LOCAL_HIREMAINDER
#endif

#if defined(INLINE) && defined(__GNUC__) && !defined(DISABLE_INLINE)
INLINE ulong /* precompute inverse of n */
get_Fl_red(ulong n)
{
  LOCAL_HIREMAINDER;
  n <<= bfffo(n);
  hiremainder = ~n;
  return divll(~0UL, n);
}
#else
INLINE ulong /* precompute inverse of n */
get_Fl_red(ulong n)
{
  ulong q, oldhi = hiremainder;
  n <<= bfffo(n);
  hiremainder = ~n;
  q = divll(~0UL, n);
  hiremainder = oldhi;
  return q;
}
#endif

INLINE ulong /* requires u1 <= n, n normalised */
divll_pre_normalized(ulong u1, ulong u0, ulong n, ulong ninv, ulong *pt_r)
{
  ulong q0, q1, r;
  LOCAL_HIREMAINDER;
  LOCAL_OVERFLOW;
  q0 = mulll(ninv, u1); q1 = hiremainder;
  q0 = addll(q0, u0);
  q1 = addllx(q1+1, u1);
  r = u0 - q1 * n;
  if (r > q0)
  {
    r += n; q1--;
  }
  if (r >= n)
  {
    r -= n; q1++;
  }
  *pt_r = r; return q1;
}

INLINE ulong /* requires u1 <= n, n normalised */
remll_pre_normalized(ulong u1, ulong u0, ulong n, ulong ninv)
{
  ulong q0, q1, r;
  LOCAL_HIREMAINDER;
  LOCAL_OVERFLOW;
  q0 = mulll(ninv, u1); q1 = hiremainder;
  q0 = addll(q0, u0);
  q1 = addllx(q1, u1);
  r = u0 - (q1 + 1) * n;
  if (r >= q0)
    r += n;
  return r < n ? r : r - n;
}

INLINE ulong /* reduce <a_hi, a_lo> mod n */
remll_pre(ulong a_hi, ulong a_lo, ulong n, ulong ninv)
{
  int norm = bfffo(n);
  int bits = BITS_IN_LONG - norm;
  ulong sn = n << norm;
  if (a_hi >= n) /* reduce a_hi first */
  {
    const ulong u1 = norm ? a_hi >> bits : 0;
    const ulong u0 = a_hi << norm;
    a_hi = remll_pre_normalized(u1, u0, sn, ninv) >> norm;
  }
  /* now reduce <a_hi, a_lo> */
  {
    const ulong u1 = ((a_hi << norm) | (norm ? a_lo >> bits: 0));
    const ulong u0 =   a_lo << norm;
    return remll_pre_normalized(u1, u0, sn, ninv) >> norm;
  }
}

#if !defined(INLINE)
extern ulong divll_pre(ulong a_lo, ulong n, ulong ninv);
#else

#if defined(__GNUC__) && !defined(DISABLE_INLINE)
#define divll_pre(a, n, ninv)                                           \
__extension__ ({                                                        \
  ulong __a = (a);                                                      \
  ulong __n = (n);                                                      \
  int norm = bfffo(__n);                                                \
  int bits = BITS_IN_LONG - norm;                                       \
  ulong r, sn = __n << norm;                                            \
  const ulong u1 = ((hiremainder << norm) | (norm ? __a >> bits: 0));   \
  const ulong u0 = __a << norm;                                         \
  const ulong q = divll_pre_normalized(u1, u0, sn, ninv, &r);           \
  hiremainder = r>>norm; q;                                             \
              })

#else /* __GNUC__ */
INLINE ulong
divll_pre(ulong a_lo, ulong n, ulong ninv)
{
  int norm = bfffo(n);
  int bits = BITS_IN_LONG - norm;
  ulong r, sn = n << norm;
  const ulong u1 = ((hiremainder << norm) | (norm ? a_lo >> bits: 0));
  const ulong u0 = a_lo << norm;
  const ulong q  = divll_pre_normalized(u1, u0, sn, ninv, &r);
  hiremainder = r>>norm; return q;
}
#endif /* __GNUC__ */

#endif
#ifdef LONG_IS_64BIT
#define __AGM_ATAN_LIMIT                 60
#define __DIVRR_GMP_LIMIT                4
#define __EXPNEWTON_LIMIT                66
#define __EXTGCD_HALFGCD_LIMIT           -1
#define __F2x_MUL_KARATSUBA_LIMIT        11
#define __F2x_MUL_MULII_LIMIT            41
#define __F2xqX_BARRETT_LIMIT            48
#define __F2xqX_DIVREM_BARRETT_LIMIT     97
#define __F2xqX_EXTGCD_LIMIT             97
#define __F2xqX_GCD_LIMIT                605
#define __F2xqX_HALFGCD_LIMIT            127
#define __F2xqX_INVBARRETT_LIMIT         50
#define __F2xqX_REM_BARRETT_LIMIT        101
#define __Flx_BARRETT2_LIMIT             31
#define __Flx_BARRETT_LIMIT              92
#define __Flx_DIVREM2_BARRETT_LIMIT      14
#define __Flx_DIVREM_BARRETT_LIMIT       161
#define __Flx_EXTGCD2_LIMIT              61
#define __Flx_EXTGCD_LIMIT               158
#define __Flx_GCD2_LIMIT                 1409
#define __Flx_GCD_LIMIT                  852
#define __Flx_HALFGCD2_LIMIT             36
#define __Flx_HALFGCD_LIMIT              120
#define __Flx_INVBARRETT2_LIMIT          22
#define __Flx_INVBARRETT_LIMIT           200
#define __Flx_MUL2_KARATSUBA_LIMIT       11
#define __Flx_MUL2_MULII_LIMIT           8
#define __Flx_MUL_KARATSUBA_LIMIT        33
#define __Flx_MUL_MULII_LIMIT            30
#define __Flx_REM2_BARRETT_LIMIT         89
#define __Flx_REM_BARRETT_LIMIT          159
#define __Flx_SQR2_KARATSUBA_LIMIT       15
#define __Flx_SQR2_SQRI_LIMIT            14
#define __Flx_SQR_KARATSUBA_LIMIT        93
#define __Flx_SQR_SQRI_LIMIT             37
#define __FlxqX_BARRETT_LIMIT            17
#define __FlxqX_DIVREM_BARRETT_LIMIT     46
#define __FlxqX_EXTGCD_LIMIT             44
#define __FlxqX_GCD_LIMIT                470
#define __FlxqX_HALFGCD_LIMIT            60
#define __FlxqX_INVBARRETT_LIMIT         22
#define __FlxqX_REM_BARRETT_LIMIT        48
#define __FpXQX_BARRETT_LIMIT            12
#define __FpXQX_DIVREM_BARRETT_LIMIT     30
#define __FpXQX_EXTGCD_LIMIT             28
#define __FpXQX_GCD_LIMIT                191
#define __FpXQX_HALFGCD_LIMIT            35
#define __FpXQX_INVBARRETT_LIMIT         40
#define __FpXQX_REM_BARRETT_LIMIT        30
#define __FpX_BARRETT_LIMIT              38
#define __FpX_DIVREM_BARRETT_LIMIT       113
#define __FpX_EXTGCD_LIMIT               87
#define __FpX_GCD_LIMIT                  406
#define __FpX_HALFGCD_LIMIT              58
#define __FpX_INVBARRETT_LIMIT           111
#define __FpX_REM_BARRETT_LIMIT          111
#define __Fp_POW_BARRETT_LIMIT           127
#define __Fp_POW_REDC_LIMIT              17
#define __GCD_HALFGCD_LIMIT              -1
#define __HALFGCD_LIMIT                  12
#define __INVMOD_GMP_LIMIT               3
#define __INVNEWTON_LIMIT                75
#define __LOGAGMCX_LIMIT                 22
#define __LOGAGM_LIMIT                   6
#define __MULII_FFT_LIMIT                -1
#define __MULII_KARATSUBA_LIMIT          -1
#define __MULRR_MULII_LIMIT              55
#define __RgX_MUL_LIMIT                  9
#define __RgX_SQR_LIMIT                  38
#define __SQRI_FFT_LIMIT                 -1
#define __SQRI_KARATSUBA_LIMIT           -1
#define __SQRR_SQRI_LIMIT                12
#else
#define __AGM_ATAN_LIMIT                 89
#define __DIVRR_GMP_LIMIT                4
#define __EXPNEWTON_LIMIT                197
#define __EXTGCD_HALFGCD_LIMIT           -1
#define __F2x_MUL_KARATSUBA_LIMIT        13
#define __F2x_MUL_MULII_LIMIT            774
#define __F2xqX_BARRETT_LIMIT            48
#define __F2xqX_DIVREM_BARRETT_LIMIT     127
#define __F2xqX_EXTGCD_LIMIT             127
#define __F2xqX_GCD_LIMIT                884
#define __F2xqX_HALFGCD_LIMIT            89
#define __F2xqX_INVBARRETT_LIMIT         40
#define __F2xqX_REM_BARRETT_LIMIT        127
#define __Flx_BARRETT2_LIMIT             52
#define __Flx_BARRETT_LIMIT              164
#define __Flx_DIVREM2_BARRETT_LIMIT      111
#define __Flx_DIVREM_BARRETT_LIMIT       470
#define __Flx_EXTGCD2_LIMIT              184
#define __Flx_EXTGCD_LIMIT               469
#define __Flx_GCD2_LIMIT                 1281
#define __Flx_GCD_LIMIT                  2817
#define __Flx_HALFGCD2_LIMIT             181
#define __Flx_HALFGCD_LIMIT              586
#define __Flx_INVBARRETT2_LIMIT          397
#define __Flx_INVBARRETT_LIMIT           501
#define __Flx_MUL2_KARATSUBA_LIMIT       9
#define __Flx_MUL2_MULII_LIMIT           8
#define __Flx_MUL_KARATSUBA_LIMIT        57
#define __Flx_MUL_MULII_LIMIT            146
#define __Flx_REM2_BARRETT_LIMIT         89
#define __Flx_REM_BARRETT_LIMIT          388
#define __Flx_SQR2_KARATSUBA_LIMIT       18
#define __Flx_SQR2_SQRI_LIMIT            20
#define __Flx_SQR_KARATSUBA_LIMIT        112
#define __Flx_SQR_SQRI_LIMIT             183
#define __FlxqX_BARRETT_LIMIT            17
#define __FlxqX_DIVREM_BARRETT_LIMIT     46
#define __FlxqX_EXTGCD_LIMIT             44
#define __FlxqX_GCD_LIMIT                1289
#define __FlxqX_HALFGCD_LIMIT            89
#define __FlxqX_INVBARRETT_LIMIT         22
#define __FlxqX_REM_BARRETT_LIMIT        48
#define __FpXQX_BARRETT_LIMIT            12
#define __FpXQX_DIVREM_BARRETT_LIMIT     30
#define __FpXQX_EXTGCD_LIMIT             28
#define __FpXQX_GCD_LIMIT                182
#define __FpXQX_HALFGCD_LIMIT            35
#define __FpXQX_INVBARRETT_LIMIT         40
#define __FpXQX_REM_BARRETT_LIMIT        30
#define __FpX_BARRETT_LIMIT              44
#define __FpX_DIVREM_BARRETT_LIMIT       116
#define __FpX_EXTGCD_LIMIT               81
#define __FpX_GCD_LIMIT                  414
#define __FpX_HALFGCD_LIMIT              55
#define __FpX_INVBARRETT_LIMIT           121
#define __FpX_REM_BARRETT_LIMIT          127
#define __Fp_POW_BARRETT_LIMIT           11
#define __Fp_POW_REDC_LIMIT              3
#define __GCD_HALFGCD_LIMIT              -1
#define __HALFGCD_LIMIT                  22
#define __INVMOD_GMP_LIMIT               3
#define __INVNEWTON_LIMIT                93
#define __LOGAGMCX_LIMIT                 32
#define __LOGAGM_LIMIT                   45
#define __MULII_FFT_LIMIT                -1
#define __MULII_KARATSUBA_LIMIT          -1
#define __MULRR_MULII_LIMIT              19
#define __RgX_MUL_LIMIT                  7
#define __RgX_SQR_LIMIT                  34
#define __SQRI_FFT_LIMIT                 -1
#define __SQRI_KARATSUBA_LIMIT           -1
#define __SQRR_SQRI_LIMIT                9
#endif
#line 2 "../src/kernel/gmp/int.h"
/* Copyright (C) 2000  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

#define int_MSW(x) ((x)+lgefint((x))-1)
/*x being a t_INT, return a pointer to the most significant word of x.*/

#define int_LSW(x) ((x)+2)
/*x being a t_INT, return a pointer to the least significant word of x.*/

#define int_precW(x) ((x)-1)
/*x pointing to a mantissa word, return the previous (less significant)
 * mantissa word.*/

#define int_nextW(x) ((x)+1)
/*x pointing to a mantissa word, return the next (more significant) mantissa
 * word.*/

#define int_W(x,l) ((x)+2+(l))
/*x being a t_INT, return a pointer to the l-th least significant word of x.*/

#define int_W_lg(x,l,lx) ((x)+2+(l))
/*x being a t_INT, return a pointer to the l-th least significant word of x,
 * assuming lgefint(x) = lx.*/

#define PARI_KERNEL_GMP
/*This macro should not be used in libpari itself.*/
#line 2 "../src/kernel/none/level1.h"
/* Copyright (C) 2000  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

/* This file defines "level 1" kernel functions.
 * These functions can be inline; they are also defined externally in
 * mpinl.c, which includes this file and never needs to be changed */

INLINE long
evallg(long x)
{
  if (x & ~LGBITS) pari_err_OVERFLOW("lg()");
  return _evallg(x);
}
INLINE long
evalvalp(long x)
{
  long v = _evalvalp(x);
  if (v & ~VALPBITS) pari_err_OVERFLOW("valp()");
  return v;
}
INLINE long
evalexpo(long x)
{
  long v = _evalexpo(x);
  if (v & ~EXPOBITS) pari_err_OVERFLOW("expo()");
  return v;
}
INLINE long
evalprecp(long x)
{
  long v = _evalprecp(x);
  if (x & ~((1UL<<(BITS_IN_LONG-VALPnumBITS))-1)) pari_err_OVERFLOW("precp()");
  return v;
}

INLINE int
varncmp(long x, long y)
{
  if (varpriority[x] < varpriority[y]) return  1;
  if (varpriority[x] > varpriority[y]) return -1;
  return 0;
}
INLINE long
varnmin(long x, long y)
{ return (varpriority[x] <= varpriority[y])? x: y; }
INLINE long
varnmax(long x, long y)
{ return (varpriority[x] >= varpriority[y])? x: y; }

/* Inhibit some area gerepile-wise: declare it to be a non recursive
 * type, of length l. Thus gerepile won't inspect the zone, just copy it.
 * For the following situation:
 *   z = cgetg(t,a); av = avma; garbage(); ltop = avma;
 *   for (i=1; i<HUGE; i++) gel(z,i) = blah();
 *   stackdummy(av,ltop);
 * loses (av-ltop) words but save a costly gerepile. */
INLINE void
stackdummy(pari_sp av, pari_sp ltop) {
  long l = ((GEN)av) - ((GEN)ltop);
  if (l > 0) {
    GEN z = (GEN)ltop;
    z[0] = evaltyp(t_VECSMALL) | evallg(l);
#ifdef DEBUG
    { long i; for (i = 1; i < l; i++) z[i] = 0; }
#endif
  }
}
INLINE void
fixlg(GEN x, long ly) {
  long lx = lg(x), l = lx - ly;
  if (l > 0)
  { /* stackdummy(x+lx, x+ly) */
    GEN z = x + ly;
    z[0] = evaltyp(t_VECSMALL) | evallg(l);
    setlg(x, ly);
#ifdef DEBUG
    { long i; for (i = 1; i < l; i++) z[i] = 0; }
#endif
  }
}
/* update lg(z) before affrr(y, z)  [ to cater for precision loss ]*/
INLINE void
affrr_fixlg(GEN y, GEN z) { fixlg(z, lg(y)); affrr(y, z); }

/*******************************************************************/
/*                                                                 */
/*                       ALLOCATE ON STACK                         */
/*                                                                 */
/*******************************************************************/
INLINE ulong
get_avma(void) { return avma; }
INLINE void
set_avma(ulong av) { avma = av; }

INLINE double
gc_double(pari_sp av, double d) { set_avma(av); return d; }
INLINE long
gc_long(pari_sp av, long s) { set_avma(av); return s; }
INLINE ulong
gc_ulong(pari_sp av, ulong s) { set_avma(av); return s; }
INLINE int
gc_bool(pari_sp av, int s) { set_avma(av); return s; }
INLINE int
gc_int(pari_sp av, int s) { set_avma(av); return s; }
INLINE GEN
gc_NULL(pari_sp av) { set_avma(av); return NULL; }
INLINE GEN
gc_const(pari_sp av, GEN x) { set_avma(av); return x; }

INLINE GEN
new_chunk(size_t x) /* x is a number of longs */
{
  GEN z = ((GEN) avma) - x;
  CHECK_CTRLC
  if (x > (avma-pari_mainstack->bot) / sizeof(long))
    new_chunk_resize(x);
  set_avma((pari_sp)z);
#ifdef MEMSTEP
  if (DEBUGMEM>1 && pari_mainstack->memused != DISABLE_MEMUSED) {
    long d = (long)pari_mainstack->memused - (long)z;
    if (labs(d) > 4*MEMSTEP)
    {
      pari_mainstack->memused = (pari_sp)z;
      err_printf("...%4.0lf Mbytes used\n",
                (pari_mainstack->top-pari_mainstack->memused)/1048576.);
    }
  }
#endif
  return z;
}

INLINE char *
stack_malloc(size_t N)
{
  long n = nchar2nlong(N);
  return (char*)new_chunk(n);
}

INLINE char *
stack_malloc_align(size_t N, long k)
{
  ulong d = ((ulong)avma) % k, e = ((ulong)N) % k;
  if (d) (void)new_chunk(d/sizeof(long));
  if (e) N += k-e;
  return (char*) new_chunk(nchar2nlong(N));
}

INLINE char *
stack_calloc(size_t N)
{
  char *p = stack_malloc(N);
  memset(p, 0, N); return p;
}

INLINE char *
stack_calloc_align(size_t N, long k)
{
  ulong d = ((ulong)avma) % k, e = ((ulong)N) % k;
  if (d) (void)new_chunk(d/sizeof(long));
  if (e) N += k-e;
  return stack_calloc(N);
}

/* cgetg(lg(x), typ(x)), set *lx. Implicit unsetisclone() */
INLINE GEN
cgetg_copy(GEN x, long *plx) {
  GEN y;
  *plx = lg(x); y = new_chunk((size_t)*plx);
  y[0] = x[0] & (TYPBITS|LGBITS); return y;
}
INLINE GEN
cgetg_block(long x, long y)
{
  GEN z = newblock((size_t)x);
  z[0] = CLONEBIT | evaltyp(y) | evallg(x);
  return z;
}
INLINE GEN
cgetg(long x, long y)
{
  GEN z = new_chunk((size_t)x);
  z[0] = evaltyp(y) | evallg(x);
  return z;
}
INLINE GEN
cgeti(long x)
{
  GEN z = new_chunk((size_t)x);
  z[0] = evaltyp(t_INT) | evallg(x);
  return z;
}
INLINE GEN
cgetipos(long x)
{
  GEN z = cgeti(x);
  z[1] = evalsigne(1) | evallgefint(x);
  return z;
}
INLINE GEN
cgetineg(long x)
{
  GEN z = cgeti(x);
  z[1] = evalsigne(-1) | evallgefint(x);
  return z;
}
INLINE GEN
cgetr_block(long x)
{
  GEN z = newblock((size_t)x);
  z[0] = CLONEBIT | evaltyp(t_REAL) | evallg(x);
  return z;
}
INLINE GEN
cgetr(long x)
{
  GEN z = new_chunk((size_t)x);
  z[0] = evaltyp(t_REAL) | evallg(x);
  return z;
}

/*******************************************************************/
/*                                                                 */
/*                     COPY, NEGATION, ABSOLUTE VALUE              */
/*                                                                 */
/*******************************************************************/
/* cannot do memcpy because sometimes x and y overlap */
INLINE GEN
leafcopy(GEN x)
{
  long lx = lg(x);
  GEN y = new_chunk(lx); /* can't use cgetg_copy, in case x,y overlap */
  while (--lx > 0) y[lx] = x[lx];
  y[0] = x[0] & (TYPBITS|LGBITS); return y;
}
INLINE GEN
icopy(GEN x)
{
  long i = lgefint(x), lx = i;
  GEN y = new_chunk(lx); /* can't use cgeti, in case x,y overlap */
  while (--i > 0) y[i] = x[i];
  y[0] = evaltyp(t_INT) | evallg(lx);
  return y;
}
INLINE GEN
icopyspec(GEN x, long nx)
{
  long i = nx+2, lx = i;
  GEN y = new_chunk(lx); /* can't use cgeti, in case x,y overlap */
  x -= 2; while (--i >= 2) y[i] = x[i];
  y[1] = evalsigne(1) | evallgefint(lx);
  y[0] = evaltyp(t_INT) | evallg(lx);
  return y;
}
INLINE GEN rcopy(GEN x) { return leafcopy(x); }
INLINE GEN mpcopy(GEN x) { return leafcopy(x); }

INLINE GEN
mpabs(GEN x) { GEN y = leafcopy(x); setabssign(y); return y; }
INLINE GEN
mpabs_shallow(GEN x) { return signe(x) < 0? mpabs(x): x; }
INLINE GEN absi(GEN x) { return mpabs(x); }
INLINE GEN absi_shallow(GEN x) { return signe(x) < 0? negi(x): x; }
INLINE GEN absr(GEN x) { return mpabs(x); }

INLINE GEN
mpneg(GEN x) { GEN y = leafcopy(x); togglesign(y); return y; }
INLINE GEN negi(GEN x) { return mpneg(x); }
INLINE GEN negr(GEN x) { return mpneg(x); }

/* negate in place */
INLINE void
togglesign(GEN x) { if (x[1] & SIGNBITS) { x[1] ^= HIGHBIT; } }
INLINE void
setabssign(GEN x) { x[1] &= ~HIGHBIT; }
/* negate in place, except universal constants */
INLINE void
togglesign_safe(GEN *px)
{
  switch(*px - gen_1) /* gen_1, gen_2, gen_m1, gen_m2 */
  {
    case 0: *px = gen_m1; break;
    case 3: *px = gen_m2;  break;
    case 6: *px = gen_1; break;
    case 9: *px = gen_2;  break;
    default: togglesign(*px);
  }
}
/* setsigne(y, signe(x)) */
INLINE void
affectsign(GEN x, GEN y)
{
  y[1] = (x[1] & SIGNBITS) | (y[1] & ~SIGNBITS);
}
/* copies sign in place, except for universal constants */
INLINE void
affectsign_safe(GEN x, GEN *py)
{
  if (((*py)[1] ^ x[1]) & HIGHBIT) togglesign_safe(py);
}
/*******************************************************************/
/*                                                                 */
/*                     GEN -> LONG, LONG -> GEN                    */
/*                                                                 */
/*******************************************************************/
/* assume x != 0, return -x as a t_INT */
INLINE GEN
utoineg(ulong x) { GEN y = cgetineg(3); y[2] = x; return y; }
/* assume x != 0, return utoi(x) */
INLINE GEN
utoipos(ulong x) { GEN y = cgetipos(3); y[2] = x; return y; }
INLINE GEN
utoi(ulong x) { return x? utoipos(x): gen_0; }
INLINE GEN
stoi(long x)
{
  if (!x) return gen_0;
  return x > 0? utoipos((ulong)x): utoineg((ulong)-x);
}

/* x 2^BIL + y */
INLINE GEN
uutoi(ulong x, ulong y)
{
  GEN z;
  if (!x) return utoi(y);
  z = cgetipos(4);
  *int_W_lg(z, 1, 4) = x;
  *int_W_lg(z, 0, 4) = y; return z;
}
/* - (x 2^BIL + y) */
INLINE GEN
uutoineg(ulong x, ulong y)
{
  GEN z;
  if (!x) return y? utoineg(y): gen_0;
  z = cgetineg(4);
  *int_W_lg(z, 1, 4) = x;
  *int_W_lg(z, 0, 4) = y; return z;
}

INLINE long
itos(GEN x)
{
  long s = signe(x);
  long u;

  if (!s) return 0;
  u = x[2];
  if (lgefint(x) > 3 || u < 0)
    pari_err_OVERFLOW("t_INT-->long assignment");
  return (s>0) ? u : -u;
}
/* as itos, but return 0 if too large. Cf is_bigint */
INLINE long
itos_or_0(GEN x) {
  long n;
  if (lgefint(x) != 3 || (n = x[2]) & HIGHBIT) return 0;
  return signe(x) > 0? n: -n;
}
INLINE ulong
itou(GEN x)
{
  switch(lgefint(x)) {
    case 2: return 0;
    case 3: return x[2];
    default:
      pari_err_OVERFLOW("t_INT-->ulong assignment");
      return 0; /* LCOV_EXCL_LINE */
  }
}

/* as itou, but return 0 if too large. Cf is_bigint */
INLINE ulong
itou_or_0(GEN x) {
  if (lgefint(x) != 3) return 0;
  return (ulong)x[2];
}

INLINE ulong
umuluu_or_0(ulong x, ulong y)
{
  ulong z;
  LOCAL_HIREMAINDER;
  z = mulll(x, y);
  return hiremainder? 0: z;
}
/* return x*y if <= n, else 0. Beware overflow */
INLINE ulong
umuluu_le(ulong x, ulong y, ulong n)
{
  ulong z;
  LOCAL_HIREMAINDER;
  z = mulll(x, y);
  return (hiremainder || z > n)? 0: z;
}

INLINE GEN
real_0_bit(long bitprec) { GEN x=cgetr(2); x[1]=evalexpo(bitprec); return x; }
INLINE GEN
real_0(long prec) { return real_0_bit(-prec2nbits(prec)); }
INLINE GEN
real_1_bit(long bit) { return real_1(nbits2prec(bit)); }
INLINE GEN
real_1(long prec) {
  GEN x = cgetr(prec);
  long i;
  x[1] = evalsigne(1) | _evalexpo(0);
  x[2] = (long)HIGHBIT; for (i=3; i<prec; i++) x[i] = 0;
  return x;
}
INLINE GEN
real_m1(long prec) {
  GEN x = cgetr(prec);
  long i;
  x[1] = evalsigne(-1) | _evalexpo(0);
  x[2] = (long)HIGHBIT; for (i=3; i<prec; i++) x[i] = 0;
  return x;
}

/* 2.^n */
INLINE GEN
real2n(long n, long prec) { GEN z = real_1(prec); setexpo(z, n); return z; }
INLINE GEN
real_m2n(long n, long prec) { GEN z = real_m1(prec); setexpo(z, n); return z; }
INLINE GEN
stor(long s, long prec) { GEN z = cgetr(prec); affsr(s,z); return z; }
INLINE GEN
utor(ulong s, long prec){ GEN z = cgetr(prec); affur(s,z); return z; }
INLINE GEN
itor(GEN x, long prec) { GEN z = cgetr(prec); affir(x,z); return z; }
INLINE GEN
rtor(GEN x, long prec) { GEN z = cgetr(prec); affrr(x,z); return z; }

INLINE ulong int_bit(GEN x, long n)
{
  long r, q = dvmdsBIL(n, &r);
  return q < lgefint(x)-2?((ulong)*int_W(x,q) >> r) & 1UL:0;
}

/*******************************************************************/
/*                                                                 */
/*                           COMPARISON                            */
/*                                                                 */
/*******************************************************************/
INLINE int
cmpss(long a, long b)
{ return a>b? 1: (a<b? -1: 0); }

INLINE int
cmpuu(ulong a, ulong b)
{ return a>b? 1: (a<b? -1: 0); }

INLINE int
cmpir(GEN x, GEN y)
{
  pari_sp av;
  GEN z;

  if (!signe(x)) return -signe(y);
  if (!signe(y))
  {
    if (expo(y) >= expi(x)) return 0;
    return signe(x);
  }
  av=avma; z = itor(x, realprec(y)); set_avma(av);
  return cmprr(z,y); /* cmprr does no memory adjustment */
}
INLINE int
cmpri(GEN x, GEN y) { return -cmpir(y,x); }
INLINE int
cmpsr(long x, GEN y)
{
  pari_sp av;
  GEN z;

  if (!x) return -signe(y);
  av=avma; z = stor(x, LOWDEFAULTPREC); set_avma(av);
  return cmprr(z,y);
}
INLINE int
cmprs(GEN x, long y) { return -cmpsr(y,x); }
/* compare x and y */
INLINE int
cmpui(ulong x, GEN y)
{
  ulong p;
  if (!x) return -signe(y);
  if (signe(y) <= 0) return 1;
  if (lgefint(y) > 3) return -1;
  p = y[2]; if (p == x) return 0;
  return p < x ? 1 : -1;
}
INLINE int
cmpiu(GEN x, ulong y) { return -cmpui(y,x); }
/* compare x and |y| */
INLINE int
abscmpui(ulong x, GEN y)
{
  long l = lgefint(y);
  ulong p;

  if (!x) return (l > 2)? -1: 0;
  if (l == 2) return 1;
  if (l > 3) return -1;
  p = y[2]; if (p == x) return 0;
  return p < x ? 1 : -1;
}
INLINE int
abscmpiu(GEN x, ulong y) { return -abscmpui(y,x); }
INLINE int
cmpsi(long x, GEN y)
{
  ulong p;

  if (!x) return -signe(y);

  if (x > 0)
  {
    if (signe(y)<=0) return 1;
    if (lgefint(y)>3) return -1;
    p = y[2]; if (p == (ulong)x) return 0;
    return p < (ulong)x ? 1 : -1;
  }

  if (signe(y)>=0) return -1;
  if (lgefint(y)>3) return 1;
  p = y[2]; if (p == (ulong)-x) return 0;
  return p < (ulong)(-x) ? -1 : 1;
}
INLINE int
cmpis(GEN x, long y) { return -cmpsi(y,x); }
INLINE int
mpcmp(GEN x, GEN y)
{
  if (typ(x)==t_INT)
    return (typ(y)==t_INT) ? cmpii(x,y) : cmpir(x,y);
  return (typ(y)==t_INT) ? -cmpir(y,x) : cmprr(x,y);
}

/* x == y ? */
INLINE int
equalui(ulong x, GEN y)
{
  if (!x) return !signe(y);
  if (signe(y) <= 0 || lgefint(y) != 3) return 0;
  return ((ulong)y[2] == (ulong)x);
}
/* x == y ? */
INLINE int
equalsi(long x, GEN y)
{
  if (!x) return !signe(y);
  if (x > 0)
  {
    if (signe(y) <= 0 || lgefint(y) != 3) return 0;
    return ((ulong)y[2] == (ulong)x);
  }
  if (signe(y) >= 0 || lgefint(y) != 3) return 0;
  return ((ulong)y[2] == (ulong)-x);
}
/* x == |y| ? */
INLINE int
absequalui(ulong x, GEN y)
{
  if (!x) return !signe(y);
  return (lgefint(y) == 3 && (ulong)y[2] == x);
}
INLINE int
absequaliu(GEN x, ulong y) { return absequalui(y,x); }
INLINE int
equalis(GEN x, long y) { return equalsi(y,x); }
INLINE int
equaliu(GEN x, ulong y) { return equalui(y,x); }

/* assume x != 0, is |x| == 2^n ? */
INLINE int
absrnz_equal2n(GEN x) {
  if ((ulong)x[2]==HIGHBIT)
  {
    long i, lx = lg(x);
    for (i = 3; i < lx; i++)
      if (x[i]) return 0;
    return 1;
  }
  return 0;
}
/* assume x != 0, is |x| == 1 ? */
INLINE int
absrnz_equal1(GEN x) { return !expo(x) && absrnz_equal2n(x); }

INLINE long
maxss(long x, long y) { return x>y?x:y; }
INLINE long
minss(long x, long y) { return x<y?x:y; }
INLINE long
minuu(ulong x, ulong y) { return x<y?x:y; }
INLINE long
maxuu(ulong x, ulong y) { return x>y?x:y; }
INLINE double
maxdd(double x, double y) { return x>y?x:y; }
INLINE double
mindd(double x, double y) { return x<y?x:y; }

/*******************************************************************/
/*                                                                 */
/*                             ADD / SUB                           */
/*                                                                 */
/*******************************************************************/
INLINE GEN
subuu(ulong x, ulong y)
{
  ulong z;
  LOCAL_OVERFLOW;
  z = subll(x, y);
  return overflow? utoineg(-z): utoi(z);
}
INLINE GEN
adduu(ulong x, ulong y) { ulong t = x+y; return uutoi((t < x), t); }

INLINE GEN
addss(long x, long y)
{
  if (!x) return stoi(y);
  if (!y) return stoi(x);
  if (x > 0) return y > 0? adduu(x,y): subuu(x, -y);

  if (y > 0) return subuu(y, -x);
  else { /* - adduu(-x, -y) */
    ulong t = (-x)+(-y); return uutoineg((t < (ulong)(-x)), t);
  }
}
INLINE GEN subss(long x, long y) { return addss(-y,x); }

INLINE GEN
subii(GEN x, GEN y)
{
  if (x==y) return gen_0; /* frequent with x = y = gen_0 */
  return addii_sign(x, signe(x), y, -signe(y));
}
INLINE GEN
addii(GEN x, GEN y) { return addii_sign(x, signe(x), y, signe(y)); }
INLINE GEN
addrr(GEN x, GEN y) { return addrr_sign(x, signe(x), y, signe(y)); }
INLINE GEN
subrr(GEN x, GEN y) { return addrr_sign(x, signe(x), y, -signe(y)); }
INLINE GEN
addir(GEN x, GEN y) { return addir_sign(x, signe(x), y, signe(y)); }
INLINE GEN
subir(GEN x, GEN y) { return addir_sign(x, signe(x), y, -signe(y)); }
INLINE GEN
subri(GEN x, GEN y) { return addir_sign(y, -signe(y), x, signe(x)); }
INLINE GEN
addsi(long x, GEN y) { return addsi_sign(x, y, signe(y)); }
INLINE GEN
addui(ulong x, GEN y) { return addui_sign(x, y, signe(y)); }
INLINE GEN
subsi(long x, GEN y) { return addsi_sign(x, y, -signe(y)); }
INLINE GEN
subui(ulong x, GEN y) { return addui_sign(x, y, -signe(y)); }

/*******************************************************************/
/*                                                                 */
/*                           MOD, REM, DIV                         */
/*                                                                 */
/*******************************************************************/
INLINE ulong mod2BIL(GEN x) { return *int_LSW(x); }
INLINE long mod64(GEN x) { return mod2BIL(x) & 63; }
INLINE long mod32(GEN x) { return mod2BIL(x) & 31; }
INLINE long mod16(GEN x) { return mod2BIL(x) & 15; }
INLINE long mod8(GEN x)  { return mod2BIL(x) & 7; }
INLINE long mod4(GEN x)  { return mod2BIL(x) & 3; }
INLINE long mod2(GEN x)  { return mod2BIL(x) & 1; }
INLINE int
mpodd(GEN x) { return signe(x) && mod2(x); }
/* x mod 2^n, n < BITS_IN_LONG */
INLINE ulong
umodi2n(GEN x, long n)
{
  long s = signe(x);
  const ulong _2n = 1UL << n;
  ulong m;
  if (!s) return 0;
  m = *int_LSW(x) & (_2n - 1);
  if (s < 0 && m) m = _2n - m;
  return m;
}
INLINE ulong Mod64(GEN x){ return umodi2n(x,6); }
INLINE ulong Mod32(GEN x){ return umodi2n(x,5); }
INLINE ulong Mod16(GEN x){ return umodi2n(x,4); }
INLINE ulong Mod8(GEN x) { return umodi2n(x,3); }
INLINE ulong Mod4(GEN x) { return umodi2n(x,2); }
INLINE ulong Mod2(GEN x) { return umodi2n(x,1); }

INLINE GEN
truedivii(GEN a,GEN b) { return truedvmdii(a,b,NULL); }
INLINE GEN
truedivis(GEN a, long b) { return truedvmdis(a,b,NULL); }
INLINE GEN
truedivsi(long a, GEN b) { return truedvmdsi(a,b,NULL); }

INLINE GEN
divii(GEN a, GEN b) { return dvmdii(a,b,NULL); }
INLINE GEN
remii(GEN a, GEN b) { return dvmdii(a,b,ONLY_REM); }

INLINE GEN
divss(long x, long y) { return stoi(x / y); }
INLINE GEN
modss(long x, long y) { return utoi(smodss(x, y)); }
INLINE GEN
remss(long x, long y) { return stoi(x % y); }
INLINE long
smodss(long x, long y)
{
  long r = x%y;
  return (r >= 0)? r: labs(y) + r;
}
INLINE ulong
umodsu(long x, ulong y)
{
  return x>=0 ? x%y: Fl_neg((-x)%y, y);
}

INLINE long
sdivss_rem(long x, long y, long *r)
{
  long q;
  LOCAL_HIREMAINDER;
  if (!y) pari_err_INV("sdivss_rem",gen_0);
  hiremainder = 0; q = divll((ulong)labs(x),(ulong)labs(y));
  if (x < 0) { hiremainder = -((long)hiremainder); q = -q; }
  if (y < 0) q = -q;
  *r = hiremainder; return q;
}
INLINE GEN
divss_rem(long x, long y, long *r) { return stoi(sdivss_rem(x,y,r)); }
INLINE ulong
udivuu_rem(ulong x, ulong y, ulong *r)
{
  if (!y) pari_err_INV("udivuu_rem",gen_0);
  *r = x % y; return x / y;
}
INLINE ulong
ceildivuu(ulong a, ulong b)
{
  ulong c;
  if (!a) return 0;
  c = a / b; return (a % b)? c+1: c;
}

INLINE ulong
uabsdivui_rem(ulong x, GEN y, ulong *r)
{
  long q, s = signe(y);
  LOCAL_HIREMAINDER;

  if (!s) pari_err_INV("uabsdivui_rem",gen_0);
  if (!x || lgefint(y)>3) { *r = x; return 0; }
  hiremainder=0; q = (long)divll(x, (ulong)y[2]);
  if (s < 0) q = -q;
  *r = hiremainder; return q;
}

/* assume d != 0 and |n| / d can be represented as an ulong.
 * Return |n|/d, set *r = |n| % d */
INLINE ulong
uabsdiviu_rem(GEN n, ulong d, ulong *r)
{
  switch(lgefint(n))
  {
    case 2: *r = 0; return 0;
    case 3:
    {
      ulong nn = n[2];
      *r = nn % d; return nn / d;
    }
    default: /* 4 */
    {
      ulong n1, n0, q;
      LOCAL_HIREMAINDER;
      n0 = *int_W(n,0);
      n1 = *int_W(n,1);
      hiremainder = n1;
      q = divll(n0, d);
      *r = hiremainder; return q;
    }
  }
}

INLINE long
sdivsi_rem(long x, GEN y, long *r)
{
  long q, s = signe(y);
  LOCAL_HIREMAINDER;

  if (!s) pari_err_INV("sdivsi_rem",gen_0);
  if (!x || lgefint(y)>3 || ((long)y[2]) < 0) { *r = x; return 0; }
  hiremainder=0; q = (long)divll(labs(x), (ulong)y[2]);
  if (x < 0) { hiremainder = -((long)hiremainder); q = -q; }
  if (s < 0) q = -q;
  *r = hiremainder; return q;
}
INLINE GEN
divsi_rem(long s, GEN y, long *r) { return stoi(sdivsi_rem(s,y,r)); }

INLINE long
sdivsi(long x, GEN y)
{
  long q, s = signe(y);

  if (!s) pari_err_INV("sdivsi",gen_0);
  if (!x || lgefint(y)>3 || ((long)y[2]) < 0) return 0;
  q = labs(x) / y[2];
  if (x < 0) q = -q;
  if (s < 0) q = -q;
  return q;
}

INLINE GEN
dvmdss(long x, long y, GEN *z)
{
  long r;
  GEN q = divss_rem(x,y, &r);
  *z = stoi(r); return q;
}
INLINE long
dvmdsBIL(long n, long *r) { *r = remsBIL(n); return divsBIL(n); }
INLINE ulong
dvmduBIL(ulong n, ulong *r) { *r = remsBIL(n); return divsBIL(n); }
INLINE GEN
dvmdsi(long x, GEN y, GEN *z)
{
  long r;
  GEN q = divsi_rem(x,y, &r);
  *z = stoi(r); return q;
}
INLINE GEN
dvmdis(GEN x, long y, GEN *z)
{
  long r;
  GEN q = divis_rem(x,y, &r);
  *z = stoi(r); return q;
}

INLINE long
smodis(GEN x, long y)
{
  pari_sp av = avma;
  long r; (void)divis_rem(x,y, &r);
  return gc_long(av, (r >= 0)? r: labs(y) + r);
}
INLINE GEN
modis(GEN x, long y) { return stoi(smodis(x,y)); }
INLINE GEN
modsi(long x, GEN y) {
  long r; (void)sdivsi_rem(x, y, &r);
  return (r >= 0)? stoi(r): addsi_sign(r, y, 1);
}

INLINE ulong
umodui(ulong x, GEN y)
{
  if (!signe(y)) pari_err_INV("umodui",gen_0);
  if (!x || lgefint(y) > 3) return x;
  return x % (ulong)y[2];
}

INLINE ulong
ugcdiu(GEN x, ulong y) { return ugcd(umodiu(x,y), y); }
INLINE ulong
ugcdui(ulong y, GEN x) { return ugcd(umodiu(x,y), y); }

INLINE GEN
remsi(long x, GEN y)
{ long r; (void)sdivsi_rem(x,y, &r); return stoi(r); }
INLINE GEN
remis(GEN x, long y)
{
  pari_sp av = avma;
  long r;
  (void)divis_rem(x,y, &r); set_avma(av); return stoi(r);
}

INLINE GEN
rdivis(GEN x, long y, long prec)
{
  GEN z = cgetr(prec);
  pari_sp av = avma;
  affrr(divrs(itor(x,prec), y),z);
  set_avma(av); return z;
}
INLINE GEN
rdivsi(long x, GEN y, long prec)
{
  GEN z = cgetr(prec);
  pari_sp av = avma;
  affrr(divsr(x, itor(y,prec)), z);
  set_avma(av); return z;
}
INLINE GEN
rdivss(long x, long y, long prec)
{
  GEN z = cgetr(prec);
  pari_sp av = avma;
  affrr(divrs(stor(x, prec), y), z);
  set_avma(av); return z;
}

INLINE void
rdiviiz(GEN x, GEN y, GEN z)
{
  long prec = realprec(z), lx = lgefint(x), ly = lgefint(y);
  if (lx == 2) { affur(0, z); return; }
  if (ly == 3)
  {
    affir(x, z); if (signe(y) < 0) togglesign(z);
    affrr(divru(z, y[2]), z);
  }
  else if (lx > prec + 1 || ly > prec + 1)
  {
    affir(x,z); affrr(divri(z, y), z);
  }
  else
  {
    long b = bit_accuracy(prec) + expi(y) - expi(x) + 1;
    GEN q = divii(b > 0? shifti(x, b): x, y);
    affir(q, z); if (b > 0) shiftr_inplace(z, -b);
  }
  set_avma((ulong)z);
}
INLINE GEN
rdivii(GEN x, GEN y, long prec)
{ GEN z = cgetr(prec); rdiviiz(x, y, z); return z; }
INLINE GEN
fractor(GEN x, long prec)
{ return rdivii(gel(x,1), gel(x,2), prec); }

INLINE int
dvdii(GEN x, GEN y)
{
  pari_sp av = avma;
  GEN r;
  if (!signe(x)) return 1;
  if (!signe(y)) return 0;
  r = remii(x,y);
  return gc_bool(av, r == gen_0);
}
INLINE int
dvdsi(long x, GEN y)
{
  if (x == 0) return 1;
  if (!signe(y)) return 0;
  if (lgefint(y) != 3) return 0;
  return x % y[2] == 0;
}
INLINE int
dvdui(ulong x, GEN y)
{
  if (x == 0) return 1;
  if (!signe(y)) return 0;
  if (lgefint(y) != 3) return 0;
  return x % y[2] == 0;
}
INLINE int
dvdis(GEN x, long y)
{ return y? smodis(x, y) == 0: signe(x) == 0; }
INLINE int
dvdiu(GEN x, ulong y)
{ return y? umodiu(x, y) == 0: signe(x) == 0; }

INLINE int
dvdisz(GEN x, long y, GEN z)
{
  const pari_sp av = avma;
  long r;
  GEN p1 = divis_rem(x,y, &r);
  set_avma(av); if (r) return 0;
  affii(p1,z); return 1;
}
INLINE int
dvdiuz(GEN x, ulong y, GEN z)
{
  const pari_sp av = avma;
  ulong r;
  GEN p1 = absdiviu_rem(x,y, &r);
  set_avma(av); if (r) return 0;
  affii(p1,z); return 1;
}
INLINE int
dvdiiz(GEN x, GEN y, GEN z)
{
  const pari_sp av=avma;
  GEN p2, p1 = dvmdii(x,y,&p2);
  if (signe(p2)) return gc_bool(av,0);
  affii(p1,z); return gc_bool(av,1);
}

INLINE ulong
remlll_pre(ulong u2, ulong u1, ulong u0, ulong n, ulong ninv)
{
  u1 = remll_pre(u2, u1, n, ninv);
  return remll_pre(u1, u0, n, ninv);
}

INLINE ulong
Fl_sqr_pre(ulong a, ulong p, ulong pi)
{
  ulong x;
  LOCAL_HIREMAINDER;
  x = mulll(a,a);
  return remll_pre(hiremainder, x, p, pi);
}

INLINE ulong
Fl_mul_pre(ulong a, ulong b, ulong p, ulong pi)
{
  ulong x;
  LOCAL_HIREMAINDER;
  x = mulll(a,b);
  return remll_pre(hiremainder, x, p, pi);
}

INLINE ulong
Fl_addmul_pre(ulong y0, ulong x0, ulong x1, ulong p, ulong pi)
{
  ulong l0, h0;
  LOCAL_HIREMAINDER;
  hiremainder = y0;
  l0 = addmul(x0, x1); h0 = hiremainder;
  return remll_pre(h0, l0, p, pi);
}

INLINE ulong
Fl_addmulmul_pre(ulong x0, ulong y0, ulong x1, ulong y1, ulong p, ulong pi)
{
  ulong l0, l1, h0, h1;
  LOCAL_OVERFLOW;
  LOCAL_HIREMAINDER;
  l0 = mulll(x0, y0); h0 = hiremainder;
  l1 = mulll(x1, y1); h1 = hiremainder;
  l0 = addll(l0, l1); h0 = addllx(h0, h1);
  return overflow ? remlll_pre(1, h0, l0, p, pi): remll_pre(h0, l0, p, pi);
}

INLINE ulong
Fl_ellj_pre(ulong a4, ulong a6, ulong p, ulong pi)
{
  /* a43 = 4 a4^3 */
  ulong a43 = Fl_double(Fl_double(
              Fl_mul_pre(a4, Fl_sqr_pre(a4, p, pi), p, pi), p), p);
  /* a62 = 27 a6^2 */
  ulong a62 = Fl_mul_pre(Fl_sqr_pre(a6, p, pi), 27 % p, p, pi);
  ulong z1 = Fl_mul_pre(a43, 1728 % p, p, pi);
  ulong z2 = Fl_add(a43, a62, p);
  return Fl_div(z1, z2, p);
}

/*******************************************************************/
/*                                                                 */
/*                        MP (INT OR REAL)                         */
/*                                                                 */
/*******************************************************************/
INLINE GEN
mptrunc(GEN x) { return typ(x)==t_INT? icopy(x): truncr(x); }
INLINE GEN
mpfloor(GEN x) { return typ(x)==t_INT? icopy(x): floorr(x); }
INLINE GEN
mpceil(GEN x) { return typ(x)==t_INT? icopy(x): ceilr(x); }
INLINE GEN
mpround(GEN x) { return typ(x) == t_INT? icopy(x): roundr(x); }

INLINE long
mpexpo(GEN x) { return typ(x) == t_INT? expi(x): expo(x); }

INLINE GEN
mpadd(GEN x, GEN y)
{
  if (typ(x)==t_INT)
    return (typ(y)==t_INT) ? addii(x,y) : addir(x,y);
  return (typ(y)==t_INT) ? addir(y,x) : addrr(x,y);
}
INLINE GEN
mpsub(GEN x, GEN y)
{
  if (typ(x)==t_INT)
    return (typ(y)==t_INT) ? subii(x,y) : subir(x,y);
  return (typ(y)==t_INT) ? subri(x,y) : subrr(x,y);
}
INLINE GEN
mpmul(GEN x, GEN y)
{
  if (typ(x)==t_INT)
    return (typ(y)==t_INT) ? mulii(x,y) : mulir(x,y);
  return (typ(y)==t_INT) ? mulir(y,x) : mulrr(x,y);
}
INLINE GEN
mpsqr(GEN x) { return (typ(x)==t_INT) ? sqri(x) : sqrr(x); }
INLINE GEN
mpdiv(GEN x, GEN y)
{
  if (typ(x)==t_INT)
    return (typ(y)==t_INT) ? divii(x,y) : divir(x,y);
  return (typ(y)==t_INT) ? divri(x,y) : divrr(x,y);
}

/*******************************************************************/
/*                                                                 */
/*                          Z/nZ, n ULONG                          */
/*                                                                 */
/*******************************************************************/
INLINE ulong
Fl_double(ulong a, ulong p)
{
  ulong res = a << 1;
  return (res >= p || res < a) ? res - p : res;
}
INLINE ulong
Fl_triple(ulong a, ulong p)
{
  ulong res = a << 1;
  if (res >= p || res < a) res -= p;
  res += a;
  return (res >= p || res < a)? res - p: res;
}
INLINE ulong
Fl_halve(ulong a, ulong p)
{
  ulong ap, ap2;
  if ((a&1UL)==0) return a>>1;
  ap = a + p; ap2 = ap>>1;
  return ap>=a ? ap2: (ap2|HIGHBIT);
}

INLINE ulong
Fl_add(ulong a, ulong b, ulong p)
{
  ulong res = a + b;
  return (res >= p || res < a) ? res - p : res;
}
INLINE ulong
Fl_neg(ulong x, ulong p) { return x ? p - x: 0; }

INLINE ulong
Fl_sub(ulong a, ulong b, ulong p)
{
  ulong res = a - b;
  return (res > a) ? res + p: res;
}

/* centerlift(u mod p) */
INLINE long
Fl_center(ulong u, ulong p, ulong ps2) { return (long) (u > ps2)? u - p: u; }

INLINE ulong
Fl_mul(ulong a, ulong b, ulong p)
{
  ulong x;
  LOCAL_HIREMAINDER;
  x = mulll(a,b);
  if (!hiremainder) return x % p;
  (void)divll(x,p); return hiremainder;
}
INLINE ulong
Fl_sqr(ulong a, ulong p)
{
  ulong x;
  LOCAL_HIREMAINDER;
  x = mulll(a,a);
  if (!hiremainder) return x % p;
  (void)divll(x,p); return hiremainder;
}
/* don't assume that p is prime: can't special case a = 0 */
INLINE ulong
Fl_div(ulong a, ulong b, ulong p)
{ return Fl_mul(a, Fl_inv(b, p), p); }

/*******************************************************************/
/*                                                                 */
/*        DEFINED FROM EXISTING ONE EXPLOITING COMMUTATIVITY       */
/*                                                                 */
/*******************************************************************/
INLINE GEN
addri(GEN x, GEN y) { return addir(y,x); }
INLINE GEN
addis(GEN x, long s) { return addsi(s,x); }
INLINE GEN
addiu(GEN x, ulong s) { return addui(s,x); }
INLINE GEN
addrs(GEN x, long s) { return addsr(s,x); }

INLINE GEN
subiu(GEN x, long y) { GEN z = subui(y, x); togglesign(z); return z; }
INLINE GEN
subis(GEN x, long y) { return addsi(-y,x); }
INLINE GEN
subrs(GEN x, long y) { return addsr(-y,x); }

INLINE GEN
mulis(GEN x, long s) { return mulsi(s,x); }
INLINE GEN
muliu(GEN x, ulong s) { return mului(s,x); }
INLINE GEN
mulru(GEN x, ulong s) { return mulur(s,x); }
INLINE GEN
mulri(GEN x, GEN s) { return mulir(s,x); }
INLINE GEN
mulrs(GEN x, long s) { return mulsr(s,x); }

/*******************************************************************/
/*                                                                 */
/*                  VALUATION, EXPONENT, SHIFTS                    */
/*                                                                 */
/*******************************************************************/
INLINE long
vali(GEN x)
{
  long i;
  GEN xp;

  if (!signe(x)) return -1;
  xp=int_LSW(x);
  for (i=0; !*xp; i++) xp=int_nextW(xp);
  return vals(*xp) + i * BITS_IN_LONG;
}

/* assume x > 0 */
INLINE long
expu(ulong x) { return (BITS_IN_LONG-1) - (long)bfffo(x); }

INLINE long
expi(GEN x)
{
  const long lx=lgefint(x);
  return lx==2? -(long)HIGHEXPOBIT: bit_accuracy(lx)-(long)bfffo(*int_MSW(x))-1;
}

INLINE GEN
shiftr(GEN x, long n)
{
  const long e = evalexpo(expo(x)+n);
  const GEN y = rcopy(x);

  if (e & ~EXPOBITS) pari_err_OVERFLOW("expo()");
  y[1] = (y[1]&~EXPOBITS) | e; return y;
}
INLINE GEN
mpshift(GEN x,long s) { return (typ(x)==t_INT)?shifti(x,s):shiftr(x,s); }

/* FIXME: adapt/use mpn_[lr]shift instead */
/* z2[imin..imax] := z1[imin..imax].f shifted left sh bits
 * (feeding f from the right). Assume sh > 0 */
INLINE void
shift_left(GEN z2, GEN z1, long imin, long imax, ulong f,  ulong sh)
{
  GEN sb = z1 + imin, se = z1 + imax, te = z2 + imax;
  ulong l, m = BITS_IN_LONG - sh, k = f >> m;
  while (se > sb) {
    l     = *se--;
    *te-- = (l << sh) | k;
    k     = l >> m;
  }
  *te = (((ulong)*se) << sh) | k;
}
/* z2[imin..imax] := f.z1[imin..imax-1] shifted right sh bits
 * (feeding f from the left). Assume sh > 0 */
INLINE void
shift_right(GEN z2, GEN z1, long imin, long imax, ulong f, ulong sh)
{
  GEN sb = z1 + imin, se = z1 + imax, tb = z2 + imin;
  ulong k, l = *sb++, m = BITS_IN_LONG - sh;
  *tb++ = (l >> sh) | (f << m);
  while (sb < se) {
    k     = l << m;
    l     = *sb++;
    *tb++ = (l >> sh) | k;
  }
}

/* Backward compatibility. Inefficient && unused */
extern ulong hiremainder;
INLINE ulong
shiftl(ulong x, ulong y)
{ hiremainder = x>>(BITS_IN_LONG-y); return (x<<y); }

INLINE ulong
shiftlr(ulong x, ulong y)
{ hiremainder = x<<(BITS_IN_LONG-y); return (x>>y); }

INLINE void
shiftr_inplace(GEN z, long d)
{
  setexpo(z, expo(z)+d);
}

/*******************************************************************/
/*                                                                 */
/*                           ASSIGNMENT                            */
/*                                                                 */
/*******************************************************************/
INLINE void
affii(GEN x, GEN y)
{
  long lx = lgefint(x);
  if (lg(y)<lx) pari_err_OVERFLOW("t_INT-->t_INT assignment");
  while (--lx) y[lx] = x[lx];
}
INLINE void
affsi(long s, GEN x)
{
  if (!s) x[1] = evalsigne(0) | evallgefint(2);
  else
  {
    if (s > 0) { x[1] = evalsigne( 1) | evallgefint(3); x[2] =  s; }
    else       { x[1] = evalsigne(-1) | evallgefint(3); x[2] = -s; }
  }
}
INLINE void
affui(ulong u, GEN x)
{
  if (!u) x[1] = evalsigne(0) | evallgefint(2);
  else  { x[1] = evalsigne(1) | evallgefint(3); x[2] = u; }
}

INLINE void
affsr(long x, GEN y)
{
  long sh, i, ly = lg(y);

  if (!x)
  {
    y[1] = evalexpo(-prec2nbits(ly));
    return;
  }
  if (x < 0) {
    x = -x; sh = bfffo(x);
    y[1] = evalsigne(-1) | _evalexpo((BITS_IN_LONG-1)-sh);
  }
  else
  {
    sh = bfffo(x);
    y[1] = evalsigne(1) | _evalexpo((BITS_IN_LONG-1)-sh);
  }
  y[2] = ((ulong)x)<<sh; for (i=3; i<ly; i++) y[i]=0;
}

INLINE void
affur(ulong x, GEN y)
{
  long sh, i, ly = lg(y);

  if (!x)
  {
    y[1] = evalexpo(-prec2nbits(ly));
    return;
  }
  sh = bfffo(x);
  y[1] = evalsigne(1) | _evalexpo((BITS_IN_LONG-1)-sh);
  y[2] = x<<sh; for (i=3; i<ly; i++) y[i] = 0;
}

INLINE void
affiz(GEN x, GEN y) { if (typ(y)==t_INT) affii(x,y); else affir(x,y); }
INLINE void
affsz(long x, GEN y) { if (typ(y)==t_INT) affsi(x,y); else affsr(x,y); }
INLINE void
mpaff(GEN x, GEN y) { if (typ(x)==t_INT) affiz(x, y); else affrr(x,y); }

/*******************************************************************/
/*                                                                 */
/*                    OPERATION + ASSIGNMENT                       */
/*                                                                 */
/*******************************************************************/

INLINE void addiiz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affii(addii(x,y),z); set_avma(av); }
INLINE void addirz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(addir(x,y),z); set_avma(av); }
INLINE void addriz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(addri(x,y),z); set_avma(av); }
INLINE void addrrz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(addrr(x,y),z); set_avma(av); }
INLINE void addsiz(long s, GEN y, GEN z)
{ pari_sp av = avma; affii(addsi(s,y),z); set_avma(av); }
INLINE void addsrz(long s, GEN y, GEN z)
{ pari_sp av = avma; affrr(addsr(s,y),z); set_avma(av); }
INLINE void addssz(long s, long y, GEN z)
{ pari_sp av = avma; affii(addss(s,y),z); set_avma(av); }

INLINE void diviiz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affii(divii(x,y),z); set_avma(av); }
INLINE void divirz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; mpaff(divir(x,y),z); set_avma(av); }
INLINE void divisz(GEN x, long y, GEN z)
{ pari_sp av = avma; affii(divis(x,y),z); set_avma(av); }
INLINE void divriz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(divri(x,y),z); set_avma(av); }
INLINE void divrrz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(divrr(x,y),z); set_avma(av); }
INLINE void divrsz(GEN y, long s, GEN z)
{ pari_sp av = avma; affrr(divrs(y,s),z); set_avma(av); }
INLINE void divsiz(long x, GEN y, GEN z)
{ long junk; affsi(sdivsi_rem(x,y,&junk), z); }
INLINE void divsrz(long s, GEN y, GEN z)
{ pari_sp av = avma; mpaff(divsr(s,y),z); set_avma(av); }
INLINE void divssz(long x, long y, GEN z)
{ affsi(x/y, z); }

INLINE void modisz(GEN y, long s, GEN z)
{ affsi(smodis(y,s),z); }
INLINE void modsiz(long s, GEN y, GEN z)
{ pari_sp av = avma; affii(modsi(s,y),z); set_avma(av); }
INLINE void modssz(long s, long y, GEN z)
{ affsi(smodss(s,y),z); }

INLINE void mpaddz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; mpaff(mpadd(x,y),z); set_avma(av); }
INLINE void mpsubz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; mpaff(mpsub(x,y),z); set_avma(av); }
INLINE void mpmulz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; mpaff(mpmul(x,y),z); set_avma(av); }

INLINE void muliiz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affii(mulii(x,y),z); set_avma(av); }
INLINE void mulirz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; mpaff(mulir(x,y),z); set_avma(av); }
INLINE void mulriz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; mpaff(mulri(x,y),z); set_avma(av); }
INLINE void mulrrz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(mulrr(x,y),z); set_avma(av); }
INLINE void mulsiz(long s, GEN y, GEN z)
{ pari_sp av = avma; affii(mulsi(s,y),z); set_avma(av); }
INLINE void mulsrz(long s, GEN y, GEN z)
{ pari_sp av = avma; mpaff(mulsr(s,y),z); set_avma(av); }
INLINE void mulssz(long s, long y, GEN z)
{ pari_sp av = avma; affii(mulss(s,y),z); set_avma(av); }

INLINE void remiiz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affii(remii(x,y),z); set_avma(av); }
INLINE void remisz(GEN y, long s, GEN z)
{ pari_sp av = avma; affii(remis(y,s),z); set_avma(av); }
INLINE void remsiz(long s, GEN y, GEN z)
{ pari_sp av = avma; affii(remsi(s,y),z); set_avma(av); }
INLINE void remssz(long s, long y, GEN z)
{ pari_sp av = avma; affii(remss(s,y),z); set_avma(av); }

INLINE void subiiz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affii(subii(x,y),z); set_avma(av); }
INLINE void subirz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(subir(x,y),z); set_avma(av); }
INLINE void subisz(GEN y, long s, GEN z)
{ pari_sp av = avma; affii(addsi(-s,y),z); set_avma(av); }
INLINE void subriz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(subri(x,y),z); set_avma(av); }
INLINE void subrrz(GEN x, GEN y, GEN z)
{ pari_sp av = avma; affrr(subrr(x,y),z); set_avma(av); }
INLINE void subrsz(GEN y, long s, GEN z)
{ pari_sp av = avma; affrr(addsr(-s,y),z); set_avma(av); }
INLINE void subsiz(long s, GEN y, GEN z)
{ pari_sp av = avma; affii(subsi(s,y),z); set_avma(av); }
INLINE void subsrz(long s, GEN y, GEN z)
{ pari_sp av = avma; affrr(subsr(s,y),z); set_avma(av); }
INLINE void subssz(long x, long y, GEN z) { addssz(x,-y,z); }

INLINE void
dvmdssz(long x, long y, GEN z, GEN t) {
  pari_sp av = avma;
  long r;
  affii(divss_rem(x,y, &r), z); set_avma(av); affsi(r,t);
}
INLINE void
dvmdsiz(long x, GEN y, GEN z, GEN t) {
  pari_sp av = avma;
  long r;
  affii(divsi_rem(x,y, &r), z); set_avma(av); affsi(r,t);
}
INLINE void
dvmdisz(GEN x, long y, GEN z, GEN t) {
  pari_sp av = avma;
  long r;
  affii(divis_rem(x,y, &r),z); set_avma(av); affsi(r,t);
}
INLINE void
dvmdiiz(GEN x, GEN y, GEN z, GEN t) {
  pari_sp av = avma;
  GEN r;
  affii(dvmdii(x,y,&r),z); affii(r,t); set_avma(av);
}
