/*
 * syserr.c
 *
 *  Created on: Jan 8, 2016
 *      Author: mramalho
 */

#undef sys_errlist
#undef sys_nerr
#undef strerror

const char *const sys_errlist[] = {
  /*  0                 */  "No error",
  /*  1 EPERM           */  "Operation not permitted",
  /*  2 ENOENT          */  "No such file or directory",
  /*  3 ESRCH           */  "No such process",
  /*  4 EINTR           */  "Interrupted function call",
  /*  5 EIO             */  "Input/output error",
  /*  6 ENXIO           */  "No such device or address",
  /*  7 E2BIG           */  "Arg list too long",
  /*  8 ENOEXEC         */  "Exec format error",
  /*  9 EBADF           */  "Bad file descriptor",
  /* 10 ECHILD          */  "No child processes",
  /* 11 EAGAIN          */  "Resource temporarily unavailable",
  /* 12 ENOMEM          */  "Not enough space",
  /* 13 EACCES          */  "Permission denied",
  /* 14 EFAULT          */  "Bad address",
  /* 15 ENOTBLK         */  "Not a block device",
  /* 16 EBUSY           */  "Resource device",
  /* 17 EEXIST          */  "File exists",
  /* 18 EXDEV           */  "Improper link",
  /* 19 ENODEV          */  "No such device",
  /* 20 ENOTDIR         */  "Not a directory",
  /* 21 EISDIR          */  "Is a directory",
  /* 22 EINVAL          */  "Invalid argument",
  /* 23 ENFILE          */  "Too many open files in system",
  /* 24 EMFILE          */  "Too many open files",
  /* 25 ENOTTY          */  "Inappropriate I/O control operation",
  /* 26 ETXTBSY         */  "Unknown error",
  /* 27 EFBIG           */  "File too large",
  /* 28 ENOSPC          */  "No space left on device",
  /* 29 ESPIPE          */  "Invalid seek",
  /* 30 EROFS           */  "Read-only file system",
  /* 31 EMLINK          */  "Too many links",
  /* 32 EPIPE           */  "Broken pipe",
  /* 33 EDOM            */  "Domain error",
  /* 34 ERANGE          */  "Result too large",
  /* 35 EUCLEAN         */  "Structure needs cleaning",
  /* 36 EDEADLK         */  "Resource deadlock avoided",
  /* 37 UNKNOWN         */  "Unknown error",
  /* 38 ENAMETOOLONG    */  "Filename too long",
  /* 39 ENOLCK          */  "No locks available",
  /* 40 ENOSYS          */  "Function not implemented",
  /* 41 ENOTEMPTY       */  "Directory not empty",
  /* 42 EILSEQ          */  "Illegal byte sequence",

  /* 43 EUNUSED43       */  "Unknown error 43",
  /* 44 EUNUSED44       */  "Unknown error 44",

  /* 45 EWOULDBLOCK     */  "Operation would block",
  /* 46 EINPROGRESS     */  "Operation now in progress",
  /* 47 EALREADY        */  "Operation already in progress",
  /* 48 ENOTSOCK        */  "Socket operation on nonsocket",
  /* 49 EDESTADDRREQ    */  "Destination address required",
  /* 50 EMSGSIZE        */  "Message too long",
  /* 51 EPROTOTYPE      */  "Protocol wrong type for socket",
  /* 52 ENOPROTOOPT     */  "Bad protocol option",
  /* 53 EPROTONOSUPPORT */  "Protocol not supported",
  /* 54 ESOCKTNOSUPPORT */  "Socket type not supported",
  /* 55 EOPNOTSUPP      */  "Operation not supported",
  /* 56 EPFNOSUPPORT    */  "Protocol family not supported",
  /* 57 EAFNOSUPPORT    */  "Address family not supported",
  /* 58 EADDRINUSE      */  "Address already in use",
  /* 59 EADDRNOTAVAIL   */  "Cannot assign requested address",
  /* 60 ENETDOWN        */  "Network is down",
  /* 61 ENETUNREACH     */  "Network is unreachable",
  /* 62 ENETRESET       */  "Network dropped connection on reset",
  /* 63 ECONNABORTED    */  "Connection aborted",
  /* 64 ECONNRESET      */  "Connection reset by peer",
  /* 65 ENOBUFS         */  "No buffer space available",
  /* 66 EISCONN         */  "Socket is already connected",
  /* 67 ENOTCONN        */  "Socket is not connected",
  /* 68 ESHUTDOWN       */  "Cannot send after socket shutdown",
  /* 69 ETOOMANYREFS    */  "Too many references",
  /* 70 ETIMEDOUT       */  "Operation timed out",
  /* 71 ECONNREFUSED    */  "Connection refused",
  /* 72 ELOOP           */  "Cannot translate name",
  /* 73 EWSNAMETOOLONG  */  "Name component or name was too long",
  /* 74 EHOSTDOWN       */  "Host is down",
  /* 75 EHOSTUNREACH    */  "No route to host",
  /* 76 EWSNOTEMPTY     */  "Cannot remove a directory that is not empty",
  /* 77 EPROCLIM        */  "Too many processes",
  /* 78 EUSERS          */  "Ran out of quota",
  /* 79 EDQUOT          */  "Ran out of disk quota",
  /* 80 ESTALE          */  "File handle reference is no longer available",
  /* 81 EREMOTE         */  "Item is not available locally",

  /* 82 EHOSTNOTFOUND   */  "Host not found",
  /* 83 ETRYAGAIN       */  "Nonauthoritative host not found",
  /* 84 ENORECOVERY     */  "A nonrecoverable error occured",
  /* 85 ENODATA         */  "Valid name, no data record of requested type",

  /* 86 EPROTO          */  "Protocol error",
  /* 87 ECHKSUM         */  "Checksum error",
  /* 88 EBADSLT         */  "Invalid slot",
  /* 89 EREMOTEIO       */  "Remote I/O error",
};


char *strerror(int errnum) {
  if (errnum < 0) errnum = -errnum;

  int sys_nerr = sizeof(sys_errlist) / sizeof(sys_errlist[0]);

  if (errnum >= sys_nerr) {
    return "Unknown error";
  } else {
    return sys_errlist[errnum];
  }
}
