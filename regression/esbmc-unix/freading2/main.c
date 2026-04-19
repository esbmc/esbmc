#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>

int main(void) {
    char buffer[32];

    FILE *fp = fmemopen(buffer, sizeof(buffer), "w+");
    if (fp == NULL)
      return -1;

    // Stream opened for update, but no read yet → not in read mode
    assert(__freading(fp) == 0);

    // Perform a write
    int ret = fputs("abc", fp);
    assert(ret >= 0);
    assert(__freading(fp) == 0);

    // Switch to reading
    rewind(fp);
    char tmp[16];
    char *res = fgets(tmp, sizeof(tmp), fp);
    assert(res != NULL);

    // Now it *is* in read mode
    assert(__freading(fp) != 0);

    fclose(fp);
    return 0;
}
