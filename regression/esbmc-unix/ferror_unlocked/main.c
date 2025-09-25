#include <assert.h>
#include <stdio.h>

int main(void) {
    char buffer[16];
    FILE *fp = fmemopen(buffer, sizeof(buffer), "r");
    if (fp == NULL)
      return -1;

    // Initially, no error should be present
    assert(ferror_unlocked(fp) == 0);

    // Attempt to write to a read-only stream (this should fail)
    int ret = fputc('X', fp);
    assert(ret == EOF);

    // Now, ferror_unlocked() should detect the error
    assert(ferror_unlocked(fp) != 0);

    fclose(fp);
    return 0;
}

