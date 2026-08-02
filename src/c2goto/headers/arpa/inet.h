#pragma once

#include <stdint.h>
#include <netinet/in.h>

__ESBMC_C_CPP_BEGIN

in_addr_t inet_addr(const char *cp);
char     *inet_ntoa(struct in_addr in);
int       inet_pton(int af, const char *src, void *dst);
const char *inet_ntop(int af, const void *src, char *dst, socklen_t size);

uint32_t htonl(uint32_t hostlong);
uint16_t htons(uint16_t hostshort);
uint32_t ntohl(uint32_t netlong);
uint16_t ntohs(uint16_t netshort);

__ESBMC_C_CPP_END
