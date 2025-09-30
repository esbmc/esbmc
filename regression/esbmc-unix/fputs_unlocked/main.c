#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *msg = "Hello, world!";
    char buffer[64];
    FILE *fp;

    // Open a memory stream (POSIX extension) for safe testing
    fp = fmemopen(buffer, sizeof(buffer), "w+");
    if (fp == NULL)
      return -1;

    // Write using fputs_unlocked
    int ret = fputs_unlocked(msg, fp);
    if (ret <= 0)
      return -1;
    assert(ret >= 0); // fputs_unlocked returns a non-negative value on success

    fclose(fp);
    return 0;
}

