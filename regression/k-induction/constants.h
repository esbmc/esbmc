#include "stubs.h"

typedef unsigned int u_int;
typedef unsigned char u_int8_t;

struct ieee80211_scan_entry {
  u_int8_t *se_rsn_ie;            /* captured RSN ie */
};

#define IEEE80211_ELEMID_RSN 200 /* fake */

/* Size of an array leader[] which is written to buf[] before it is
 * overflowed by the ie[] array. */
#define LEADERSZ 1

/* We first write the "leader" to buf[], and then write from the "ie"
 * array. buf[] has to be bigger than LEADERSZ by at least 2. */
#define BUFSZ BASE_SZ + LEADERSZ + 3

/* Just has to be big enough to overflow buf[]
 * Note that for each byte in ie[], two bytes are written to buf[] in
 * encode_ie() */
#define IESZ BUFSZ - LEADERSZ

typedef int NSS_STATUS;

/* Size of overflowed buffer. */
#define FSTRING_LEN BASE_SZ /* originally 256 */
typedef char fstring[FSTRING_LEN];

/* Size of input buffer. */
#define INSZ (FSTRING_LEN+2)

// Destination buffer.
#define BUF BASE_SZ

// Source buffers. Make each big enough that the size checks in the OK
// versions are necessary to ensure safety.
#define GECOS BASE_SZ + 2
#define LOGIN BASE_SZ + 2

#define EXPRESSION_LENGTH BASE_SZ
#define NEEDLE "EX"
#define NEEDLE_SZ 2

/* Enough to fill a buffer of size EXPRESSION_LENGTH, enough to
 * contain the needle, and enough to overflow the buffer. */
#define LINE_LENGTH EXPRESSION_LENGTH + NEEDLE_SZ + 4

/* Size of buffer being overflowed.
 * Ensure that SUN_PATH_SZ - 1 is non-negative */
#define SUN_PATH_SZ BASE_SZ + 1/* originally 108 */

/* Size of input buffer. */
#define FILENAME_SZ SUN_PATH_SZ + 2  /* originally 1024 */

struct sockaddr_un
{
  char sun_path[SUN_PATH_SZ];         /* Path name.  */
};
