#include <stdint.h>

#undef htonl
#undef htons
#undef ntohl
#undef ntohs

uint32_t htonl(uint32_t hostlong)
{
  if (__ESBMC_is_little_endian())
    return __builtin_bswap32(hostlong);
  return hostlong;
}

uint16_t htons(uint16_t hostshort)
{
  if (__ESBMC_is_little_endian())
    return __builtin_bswap16(hostshort);
  return hostshort;
}

uint32_t ntohl(uint32_t netlong)
{
  if (__ESBMC_is_little_endian())
    return __builtin_bswap32(netlong);
  return netlong;
}

uint16_t ntohs(uint16_t netshort)
{
  if (__ESBMC_is_little_endian())
    return __builtin_bswap16(netshort);
  return netshort;
}
