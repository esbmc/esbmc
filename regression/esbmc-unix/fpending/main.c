#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    char buffer[128];
    FILE *fp = fmemopen(buffer, sizeof(buffer), "w+");
    if (fp == NULL)
      return -1;

    // At the start, nothing is pending
    assert(__fpending(fp) == 0);

    // Write some data, but don't flush yet
    fputs("hello", fp);

    size_t pending = __fpending(fp);
    assert(pending > 0); // something should be buffered

    // Flushing should empty the buffer
    fflush(fp);
    assert(__fpending(fp) == 0);

    // Writing again
    fputs("world", fp);
    pending = __fpending(fp);
    assert(pending > 0);

    fclose(fp);
    return 0;
}

