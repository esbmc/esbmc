#pragma once

#include <__esbmc/stddefs.h>
#include <stdint.h>
#include <stddef.h>

__ESBMC_C_CPP_BEGIN

/* Guard against the typedef the C library headers expose under their own
 * macro (glibc uses __ssize_t_defined) so including <sys/types.h> or
 * <unistd.h> alongside this header does not trigger a redefinition. */
#if !defined(_SSIZE_T_DEFINED) && !defined(__ssize_t_defined)
#define _SSIZE_T_DEFINED
#define __ssize_t_defined
typedef long ssize_t;
#endif

/* Address families */
#define AF_UNSPEC    0
#define AF_UNIX      1
#define AF_LOCAL     1
#define AF_INET      2
#define AF_INET6    10

/* Protocol families (aliases for AF_*) */
#define PF_UNSPEC    AF_UNSPEC
#define PF_UNIX      AF_UNIX
#define PF_LOCAL     AF_LOCAL
#define PF_INET      AF_INET
#define PF_INET6     AF_INET6

/* Socket types */
#define SOCK_STREAM    1
#define SOCK_DGRAM     2
#define SOCK_RAW       3

/* Socket options / levels */
#define SOL_SOCKET     1
#define SO_REUSEADDR   2
#define SO_KEEPALIVE   9
#define SO_RCVBUF     8
#define SO_SNDBUF     7

/* Shutdown modes */
#define SHUT_RD    0
#define SHUT_WR    1
#define SHUT_RDWR  2

/* Message flags */
#define MSG_DONTWAIT  0x40
#define MSG_NOSIGNAL  0x4000
#define MSG_PEEK      0x02

/* Socket address length type */
typedef unsigned int socklen_t;

/* Generic socket address */
typedef unsigned short sa_family_t;

struct sockaddr {
    sa_family_t sa_family; // cppcheck-suppress unusedStructMember
    char        sa_data[14]; // cppcheck-suppress unusedStructMember
};

struct sockaddr_storage {
    sa_family_t ss_family; // cppcheck-suppress unusedStructMember
    char        __ss_pad1[6]; // cppcheck-suppress unusedStructMember
    int64_t     __ss_align; // cppcheck-suppress unusedStructMember
    char        __ss_pad2[112]; // cppcheck-suppress unusedStructMember
};

/* I/O vector for scatter/gather I/O */
struct iovec {
    void  *iov_base; // cppcheck-suppress unusedStructMember
    size_t iov_len; // cppcheck-suppress unusedStructMember
};

struct msghdr {
    void         *msg_name; // cppcheck-suppress unusedStructMember
    socklen_t     msg_namelen; // cppcheck-suppress unusedStructMember
    struct iovec *msg_iov; // cppcheck-suppress unusedStructMember
    size_t        msg_iovlen; // cppcheck-suppress unusedStructMember
    void         *msg_control; // cppcheck-suppress unusedStructMember
    size_t        msg_controllen; // cppcheck-suppress unusedStructMember
    int           msg_flags; // cppcheck-suppress unusedStructMember
};

/* Function declarations */
int    socket(int domain, int type, int protocol);
int    bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int    listen(int sockfd, int backlog);
int    accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
int    connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

ssize_t send(int sockfd, const void *buf, size_t len, int flags);
ssize_t recv(int sockfd, void *buf, size_t len, int flags);
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen);
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen);

int    setsockopt(int sockfd, int level, int optname,
                  const void *optval, socklen_t optlen);
int    getsockopt(int sockfd, int level, int optname,
                  void *optval, socklen_t *optlen);
int    getsockname(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
int    getpeername(int sockfd, struct sockaddr *addr, socklen_t *addrlen);

int    shutdown(int sockfd, int how);

__ESBMC_C_CPP_END
