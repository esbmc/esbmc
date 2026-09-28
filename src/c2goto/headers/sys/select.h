#pragma once

#include <sys/socket.h>

__ESBMC_C_CPP_BEGIN

/* Number of file descriptors that fit in fd_set */
#define FD_SETSIZE 1024

typedef struct {
    unsigned long fds_bits[FD_SETSIZE / (8 * sizeof(unsigned long))];
} fd_set;

#define FD_ZERO(set) \
    do { \
        unsigned int __i; \
        for (__i = 0; __i < sizeof(fd_set) / sizeof(unsigned long); __i++) \
            ((fd_set *)(set))->fds_bits[__i] = 0; \
    } while (0)

#define FD_SET(fd, set) \
    ((fd_set *)(set))->fds_bits[(fd) / (8 * sizeof(unsigned long))] \
        |= (1UL << ((fd) % (8 * sizeof(unsigned long))))

#define FD_CLR(fd, set) \
    ((fd_set *)(set))->fds_bits[(fd) / (8 * sizeof(unsigned long))] \
        &= ~(1UL << ((fd) % (8 * sizeof(unsigned long))))

#define FD_ISSET(fd, set) \
    (((fd_set *)(set))->fds_bits[(fd) / (8 * sizeof(unsigned long))] \
        & (1UL << ((fd) % (8 * sizeof(unsigned long)))))

/* Guard with the macros both libcs use -- glibc's __timeval_defined
 * (bits/types/struct_timeval.h) and Apple's _STRUCT_TIMEVAL
 * (sys/_types/_timeval.h). Defining theirs makes any SDK header that would
 * define timeval skip it; honouring theirs stops us redefining it when a
 * system header we do not shadow got there first. */
#if !defined(__timeval_defined) && !defined(_STRUCT_TIMEVAL)
#define __timeval_defined 1
#define _STRUCT_TIMEVAL struct timeval
struct timeval {
    long tv_sec; // cppcheck-suppress unusedStructMember
    long tv_usec; // cppcheck-suppress unusedStructMember
};
#endif

int select(int nfds, fd_set *readfds, fd_set *writefds,
           fd_set *exceptfds, struct timeval *timeout);

/* poll */
struct pollfd {
    int   fd; // cppcheck-suppress unusedStructMember
    short events; // cppcheck-suppress unusedStructMember
    short revents; // cppcheck-suppress unusedStructMember
};

#define POLLIN     0x0001
#define POLLPRI    0x0002
#define POLLOUT    0x0004
#define POLLERR    0x0008
#define POLLHUP    0x0010
#define POLLNVAL   0x0020
#define POLLRDNORM 0x0040
#define POLLWRNORM 0x0100

int poll(struct pollfd *fds, unsigned long nfds, int timeout);

__ESBMC_C_CPP_END
