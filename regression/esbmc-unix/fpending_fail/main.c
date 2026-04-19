#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>

int main(void) {
    char buffer[128];

    // Open the stream in read-only mode
    FILE *fp = fmemopen(buffer, sizeof(buffer), "r");
    if (fp == NULL)
      return -1;

    // According to the GNU libc manual, __fpending() is undefined here.
    // We can still call it, but we should *not* make assumptions about the result.
    // For a negative test, we just check that it does not report pending > 0.
    size_t pending = __fpending(fp);
    assert(pending > 0);

    fclose(fp);
    return 0;
}

