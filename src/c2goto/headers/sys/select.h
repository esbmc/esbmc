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

struct timeval {
    long tv_sec;
    long tv_usec;
};

int select(int nfds, fd_set *readfds, fd_set *writefds,
           fd_set *exceptfds, struct timeval *timeout);

/* poll */
struct pollfd {
    int   fd;
    short events;
    short revents;
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
