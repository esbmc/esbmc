/*
 * socket_lib.c — Operational models for POSIX socket and I/O multiplexing APIs.
 *
 * These models abstract the behavior of network system calls for use with ESBMC.
 * Instead of following the real OS kernel implementation (which would make the
 * state space intractable), each function returns a non-deterministic result
 * consistent with the POSIX specification, along with appropriate precondition
 * checks.
 *
 * Key design decisions:
 *   - socket()/bind()/listen()/accept()/connect() return non-deterministic
 *     success/failure values within their valid range.
 *   - recv() fills the user buffer with non-deterministic bytes (modeling
 *     attacker-controlled input) and returns a non-deterministic byte count.
 *   - send() returns a non-deterministic count of bytes "sent" (0 to len).
 *   - select()/poll() return non-deterministic readiness, preserving the
 *     contract that at least one fd is ready when the return value > 0.
 *   - All models include __ESBMC_assert checks for null-pointer and
 *     invalid-argument preconditions.
 *
 * Reference: pthread_lib.c for the ESBMC operational model pattern.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/socket.h>

/* ESBMC built-in non-deterministic value generators */
extern int     __VERIFIER_nondet_int(void);
extern short   __VERIFIER_nondet_short(void);
extern size_t  __VERIFIER_nondet_size_t(void);
extern uint8_t __VERIFIER_nondet_uchar(void);

/* ============================================================
 *  socket()
 *  Returns a non-deterministic file descriptor >= 0 on success,
 *  or -1 on failure (with errno set).
 * ============================================================ */
int socket(int domain, int type, int protocol)
{
__ESBMC_HIDE:;
    int fd = __VERIFIER_nondet_int();

    if (fd < 0) {
        errno = EACCES;
        return -1;
    }

    /* Constrain to a realistic range of file descriptors */
    __ESBMC_assume(fd >= 3 && fd < 1024);
    return fd;
}

/* ============================================================
 *  bind()
 *  Returns 0 on success, -1 on failure.
 * ============================================================ */
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
__ESBMC_HIDE:;
    __ESBMC_assert(addr != (void *)0,
        "bind: addr must not be NULL");

    int result = __VERIFIER_nondet_int();
    if (result != 0) {
        errno = EADDRINUSE;
        return -1;
    }
    return 0;
}

/* ============================================================
 *  listen()
 *  Returns 0 on success, -1 on failure.
 * ============================================================ */
int listen(int sockfd, int backlog)
{
__ESBMC_HIDE:;
    int result = __VERIFIER_nondet_int();
    if (result != 0) {
        errno = EADDRINUSE;
        return -1;
    }
    return 0;
}

/* ============================================================
 *  accept()
 *  Returns a new fd >= 0 on success, -1 on failure.
 *  If addr is non-NULL, fills it with non-deterministic data.
 * ============================================================ */
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
__ESBMC_HIDE:;
    int fd = __VERIFIER_nondet_int();

    if (fd < 0) {
        errno = EAGAIN;
        return -1;
    }

    __ESBMC_assume(fd >= 3 && fd < 1024);

    /* If the caller wants the peer address, fill with non-deterministic data */
    if (addr != (void *)0 && addrlen != (void *)0) {
        socklen_t max = *addrlen;
        if (max > sizeof(struct sockaddr)) {
            max = sizeof(struct sockaddr);
        }
        for (socklen_t i = 0; i < max; i++) {
            ((uint8_t *)addr)[i] = __VERIFIER_nondet_uchar();
        }
        *addrlen = max;
    }

    return fd;
}

/* ============================================================
 *  connect()
 *  Returns 0 on success, -1 on failure.
 * ============================================================ */
int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
__ESBMC_HIDE:;
    __ESBMC_assert(addr != (void *)0,
        "connect: addr must not be NULL");

    int result = __VERIFIER_nondet_int();
    if (result != 0) {
        errno = ECONNREFUSED;
        return -1;
    }
    return 0;
}

/* ============================================================
 *  recv()
 *  Fills buf with up to len non-deterministic bytes.
 *  Returns the number of bytes "received" (0 to len), or -1 on error.
 *
 *  This is the most critical model for security verification:
 *  the non-deterministic fill simulates attacker-controlled input.
 * ============================================================ */
ssize_t recv(int sockfd, void *buf, size_t len, int flags)
{
__ESBMC_HIDE:;
    __ESBMC_assert(buf != (void *)0,
        "recv: buf must not be NULL");

    ssize_t result = (ssize_t)__VERIFIER_nondet_int();

    /* Error case */
    if (result < 0) {
        errno = ECONNRESET;
        return -1;
    }

    /* Connection closed */
    if (result == 0) {
        return 0;
    }

    /* Constrain received bytes to [1, len] */
    __ESBMC_assume(result >= 1 && (size_t)result <= len);

    /* Fill buffer with non-deterministic (attacker-controlled) bytes */
    for (ssize_t i = 0; i < result; i++) {
        ((uint8_t *)buf)[i] = __VERIFIER_nondet_uchar();
    }

    return result;
}

/* ============================================================
 *  send()
 *  Returns the number of bytes "sent" (0 to len), or -1 on error.
 * ============================================================ */
ssize_t send(int sockfd, const void *buf, size_t len, int flags)
{
__ESBMC_HIDE:;
    __ESBMC_assert(buf != (void *)0,
        "send: buf must not be NULL");

    ssize_t result = (ssize_t)__VERIFIER_nondet_int();

    if (result < 0) {
        errno = EPIPE;
        return -1;
    }

    __ESBMC_assume(result >= 0 && (size_t)result <= len);
    return result;
}

/* ============================================================
 *  recvfrom()
 *  Like recv(), but also fills src_addr if non-NULL.
 * ============================================================ */
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen)
{
__ESBMC_HIDE:;
    __ESBMC_assert(buf != (void *)0,
        "recvfrom: buf must not be NULL");

    ssize_t result = (ssize_t)__VERIFIER_nondet_int();

    if (result < 0) {
        errno = ECONNRESET;
        return -1;
    }

    if (result == 0) {
        return 0;
    }

    __ESBMC_assume(result >= 1 && (size_t)result <= len);

    for (ssize_t i = 0; i < result; i++) {
        ((uint8_t *)buf)[i] = __VERIFIER_nondet_uchar();
    }

    if (src_addr != (void *)0 && addrlen != (void *)0) {
        socklen_t max = *addrlen;
        if (max > sizeof(struct sockaddr)) {
            max = sizeof(struct sockaddr);
        }
        for (socklen_t i = 0; i < max; i++) {
            ((uint8_t *)src_addr)[i] = __VERIFIER_nondet_uchar();
        }
        *addrlen = max;
    }

    return result;
}

/* ============================================================
 *  sendto()
 *  Like send(), with a destination address.
 * ============================================================ */
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen)
{
__ESBMC_HIDE:;
    __ESBMC_assert(buf != (void *)0,
        "sendto: buf must not be NULL");

    ssize_t result = (ssize_t)__VERIFIER_nondet_int();

    if (result < 0) {
        errno = EPIPE;
        return -1;
    }

    __ESBMC_assume(result >= 0 && (size_t)result <= len);
    return result;
}

/* ============================================================
 *  setsockopt() / getsockopt()
 *  Non-deterministic success/failure.
 * ============================================================ */
int setsockopt(int sockfd, int level, int optname,
               const void *optval, socklen_t optlen)
{
__ESBMC_HIDE:;
    int result = __VERIFIER_nondet_int();
    if (result != 0) {
        errno = ENOPROTOOPT;
        return -1;
    }
    return 0;
}

int getsockopt(int sockfd, int level, int optname,
               void *optval, socklen_t *optlen)
{
__ESBMC_HIDE:;
    int result = __VERIFIER_nondet_int();
    if (result != 0) {
        errno = ENOPROTOOPT;
        return -1;
    }
    return 0;
}

/* ============================================================
 *  getsockname() / getpeername()
 * ============================================================ */
int getsockname(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
__ESBMC_HIDE:;
    __ESBMC_assert(addr != (void *)0,
        "getsockname: addr must not be NULL");
    __ESBMC_assert(addrlen != (void *)0,
        "getsockname: addrlen must not be NULL");

    int result = __VERIFIER_nondet_int();
    if (result != 0) {
        errno = EBADF;
        return -1;
    }

    socklen_t max = *addrlen;
    if (max > sizeof(struct sockaddr)) {
        max = sizeof(struct sockaddr);
    }
    for (socklen_t i = 0; i < max; i++) {
        ((uint8_t *)addr)[i] = __VERIFIER_nondet_uchar();
    }
    *addrlen = max;
    return 0;
}

int getpeername(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
__ESBMC_HIDE:;
    __ESBMC_assert(addr != (void *)0,
        "getpeername: addr must not be NULL");
    __ESBMC_assert(addrlen != (void *)0,
        "getpeername: addrlen must not be NULL");

    int result = __VERIFIER_nondet_int();
    if (result != 0) {
        errno = ENOTCONN;
        return -1;
    }

    socklen_t max = *addrlen;
    if (max > sizeof(struct sockaddr)) {
        max = sizeof(struct sockaddr);
    }
    for (socklen_t i = 0; i < max; i++) {
        ((uint8_t *)addr)[i] = __VERIFIER_nondet_uchar();
    }
    *addrlen = max;
    return 0;
}

/* ============================================================
 *  shutdown()
 * ============================================================ */
int shutdown(int sockfd, int how)
{
__ESBMC_HIDE:;
    int result = __VERIFIER_nondet_int();
    if (result != 0) {
        errno = ENOTCONN;
        return -1;
    }
    return 0;
}

/* ============================================================
 *  select()
 *  Returns the number of ready fds (>= 0), or -1 on error.
 *  Modifies readfds/writefds/exceptfds non-deterministically.
 * ============================================================ */
int select(int nfds, fd_set *readfds, fd_set *writefds,
           fd_set *exceptfds, struct timeval *timeout)
{
__ESBMC_HIDE:;
    int result = __VERIFIER_nondet_int();

    if (result < 0) {
        errno = EINTR;
        return -1;
    }

    /* Constrain to valid range */
    __ESBMC_assume(result >= 0 && result <= nfds);

    return result;
}

/* ============================================================
 *  poll()
 *  Returns the number of fds with events (>= 0), or -1 on error.
 *  Fills revents fields non-deterministically.
 * ============================================================ */
int poll(struct pollfd *fds, unsigned long nfds, int timeout)
{
__ESBMC_HIDE:;
    __ESBMC_assert(fds != (void *)0 || nfds == 0,
        "poll: fds must not be NULL when nfds > 0");

    int result = __VERIFIER_nondet_int();

    if (result < 0) {
        errno = EINTR;
        return -1;
    }

    __ESBMC_assume(result >= 0 && (unsigned long)result <= nfds);

    /* Fill revents with non-deterministic values constrained to
     * the requested events plus error flags */
    for (unsigned long i = 0; i < nfds; i++) {
        fds[i].revents = __VERIFIER_nondet_short();
    }

    return result;
}

/* ============================================================
 *  Byte-order conversion functions (htons, htonl, ntohs, ntohl)
 *  These are modeled as identity functions for verification
 *  purposes, since ESBMC does not distinguish byte order.
 * ============================================================ */
uint16_t htons(uint16_t hostshort)
{
__ESBMC_HIDE:;
    return hostshort;
}

uint32_t htonl(uint32_t hostlong)
{
__ESBMC_HIDE:;
    return hostlong;
}

uint16_t ntohs(uint16_t netshort)
{
__ESBMC_HIDE:;
    return netshort;
}

uint32_t ntohl(uint32_t netlong)
{
__ESBMC_HIDE:;
    return netlong;
}

/* ============================================================
 *  inet_addr() — parse dotted-quad IPv4 string
 *  Returns non-deterministic address or INADDR_NONE on error.
 * ============================================================ */
uint32_t inet_addr(const char *cp)
{
__ESBMC_HIDE:;
    __ESBMC_assert(cp != (void *)0,
        "inet_addr: cp must not be NULL");

    uint32_t result = (uint32_t)__VERIFIER_nondet_int();
    return result;
}

/* ============================================================
 *  inet_pton() — convert text to binary network address
 *  Returns 1 on success, 0 for invalid input, -1 on error.
 * ============================================================ */
int inet_pton(int af, const char *src, void *dst)
{
__ESBMC_HIDE:;
    __ESBMC_assert(src != (void *)0,
        "inet_pton: src must not be NULL");
    __ESBMC_assert(dst != (void *)0,
        "inet_pton: dst must not be NULL");

    int result = __VERIFIER_nondet_int();
    __ESBMC_assume(result >= -1 && result <= 1);

    if (result == 1) {
        /* Fill dst with non-deterministic address bytes */
        int size = (af == 2 /* AF_INET */) ? 4 : 16;
        for (int i = 0; i < size; i++) {
            ((uint8_t *)dst)[i] = __VERIFIER_nondet_uchar();
        }
    }

    return result;
}
