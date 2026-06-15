#pragma once

#include <stdint.h>
#include <sys/socket.h>

__ESBMC_C_CPP_BEGIN

typedef uint32_t in_addr_t;
typedef uint16_t in_port_t;

struct in_addr {
    in_addr_t s_addr;
};

/* IPv4 socket address */
struct sockaddr_in {
    sa_family_t    sin_family;
    in_port_t      sin_port;
    struct in_addr sin_addr;
    unsigned char  sin_zero[8];
};

/* IPv6 address */
struct in6_addr {
    union {
        uint8_t  s6_addr[16];
        uint16_t s6_addr16[8];
        uint32_t s6_addr32[4];
    };
};

/* IPv6 socket address */
struct sockaddr_in6 {
    sa_family_t     sin6_family;
    in_port_t       sin6_port;
    uint32_t        sin6_flowinfo;
    struct in6_addr sin6_addr;
    uint32_t        sin6_scope_id;
};

#define INADDR_ANY       ((in_addr_t) 0x00000000)
#define INADDR_BROADCAST ((in_addr_t) 0xffffffff)
#define INADDR_LOOPBACK  ((in_addr_t) 0x7f000001)

#define IPPROTO_IP    0
#define IPPROTO_TCP   6
#define IPPROTO_UDP  17

__ESBMC_C_CPP_END
