// Native libc sweeps over the SAME domains the ESBMC harnesses cover, so the
// times are comparable. Each mode checks the same property the harness proves.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/fixed_point/fx_bits.h"
namespace fx = LIBC_NAMESPACE::fixed_point;

int main(int argc, char **argv){
  const char *mode = argc>1?argv[1]:"";
  long viol=0, checked=0;
  if(!strcmp(mode,"sqrt_uhr")){           // u0.8, 2^8 inputs
    for(unsigned x=0;x<256u;x++){
      unsigned short _Fract v; uint8_t b=(uint8_t)x; memcpy(&v,&b,1);
      unsigned short _Fract r=fx::sqrt(v); uint8_t rb; memcpy(&rb,&r,1);
      unsigned long long lo=(unsigned long long)rb*rb, xs=(unsigned long long)x<<8;
      checked++; if(!(lo<=xs)) viol++;
    }
  } else if(!strcmp(mode,"sqrt_ur")){     // u0.16, 2^16 inputs
    for(unsigned x=0;x<65536u;x++){
      unsigned _Fract v; uint16_t b=(uint16_t)x; memcpy(&v,&b,2);
      unsigned _Fract r=fx::sqrt(v); uint16_t rb; memcpy(&rb,&r,2);
      unsigned long long lo=(unsigned long long)rb*rb, xs=(unsigned long long)x<<16;
      checked++; if(!(lo<=xs)) viol++;
    }
  } else if(!strcmp(mode,"sqrt_ulr")){    // u0.32, 2^32 inputs -- EXHAUSTIVE
    for(unsigned long long x=0;x<4294967296ull;x++){
      unsigned long _Fract v; uint32_t b=(uint32_t)x; memcpy(&v,&b,4);
      unsigned long _Fract r=fx::sqrt(v); uint32_t rb; memcpy(&rb,&r,4);
      __uint128_t lo=(__uint128_t)rb*rb, xs=(__uint128_t)x<<32;
      checked++; if(!(lo<=xs)) viol++;
    }
  } else if(!strcmp(mode,"isqrt_uhk")){   // u8.8 from unsigned short, 2^16
    for(unsigned n=0;n<65536u;n++){
      unsigned short _Accum r=fx::isqrt((unsigned short)n);
      uint16_t rb; memcpy(&rb,&r,2);
      __uint128_t ns=(__uint128_t)n<<16;
      __uint128_t mx=(__uint128_t)65535u*65535u;
      if(ns>mx) continue;
      __uint128_t up=((__uint128_t)rb+1)*((__uint128_t)rb+1);
      checked++; if(!(ns<up)) viol++;
    }
  } else if(!strcmp(mode,"isqrt_uk")){    // u16.16 from unsigned int, 2^32 EXHAUSTIVE
    for(unsigned long long n=0;n<4294967296ull;n++){
      unsigned _Accum r=fx::isqrt((unsigned)n);
      uint32_t rb; memcpy(&rb,&r,4);
      __uint128_t ns=(__uint128_t)n<<32;
      __uint128_t mx=(__uint128_t)4294967295u*4294967295u;
      if(ns>mx) continue;
      __uint128_t up=((__uint128_t)rb+1)*((__uint128_t)rb+1);
      checked++; if(!(ns<up)) viol++;
    }
  } else { fprintf(stderr,"unknown mode\n"); return 2; }
  printf("%s: checked %ld, violations %ld\n", mode, checked, viol);
  return 0;
}
