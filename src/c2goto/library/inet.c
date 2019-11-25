#include <arpa/inet.h>
#include <stdint.h>

#undef htonl
#undef htons
#undef ntohl
#undef ntohs

in_addr_t __VERIFIER_nondet_in_addr_t();
in_addr_t inet_addr(const char *cp)
{
__ESBMC_HIDE:;
  (void)*cp;
  return __VERIFIER_nondet_in_addr_t();
}

int inet_aton(const char *cp, struct in_addr *pin)
{
__ESBMC_HIDE:;
  (void)*cp;
  (void)*pin;
  return __VERIFIER_nondet_int();
}

in_addr_t inet_network(const char *cp)
{
__CPROVER_HIDE:;
  (void)*cp;
  return __VERIFIER_nondet_in_addr_t();
}

uint32_t htonl(uint32_t hostlong)
{
  if(__ESBMC_is_little_endian())
    return __builtin_bswap32(hostlong);
  return hostlong;
}

uint16_t htons(uint16_t hostshort)
{
  if(__ESBMC_is_little_endian())
    return __builtin_bswap16(hostshort);
  return hostshort;
}

uint32_t ntohl(uint32_t netlong)
{
  if(__ESBMC_is_little_endian())
    return __builtin_bswap32(netlong);
  return netlong;
}

uint16_t ntohs(uint16_t netshort)
{
  if(__ESBMC_is_little_endian())
    return __builtin_bswap16(netshort);
  return netshort;
}
