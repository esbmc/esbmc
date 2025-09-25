#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    char buffer[32];
    FILE *fp = fmemopen(buffer, sizeof(buffer), "w+");
    if (fp == NULL)
      return -1;

    // No error at start
    assert(ferror_unlocked(fp) == 0);

    // Write some data (valid operation)
    int ret = fputs("hello", fp);
    assert(ret >= 0);

    // Flush should succeed without error
    fflush(fp);
    assert(ferror_unlocked(fp) == 0);

    // Rewind and read back safely
    rewind(fp);
    char readback[32];
    char *res = fgets(readback, sizeof(readback), fp);
    assert(res != NULL);

    // Still no error
    assert(ferror_unlocked(fp) == 0);

    fclose(fp);
    return 0;
}

