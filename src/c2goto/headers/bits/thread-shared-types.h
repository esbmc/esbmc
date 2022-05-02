/* Glibc private header, included by
 * - <threads.h>, standard C11 header, provided by Glibc, included by
 *   - <c++/v1/__threading_support>, LLVM libc++ private header, included by
 *     - <c++/v1/mutex>
 *       <c++/v1/atomic>
 *       <c++/v1/semaphore>
 *       <c++/v1/thread>
 *       LLVM libc++ public headers
 * - <bits/pthreadtypes.h>, provided by Glibc, see there
 *
 * Just redirect to our <pthread.h>
 */

#include <pthread.h>
