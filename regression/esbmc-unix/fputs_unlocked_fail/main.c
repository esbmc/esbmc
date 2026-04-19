#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *msg = "Hello, world!";
    char buffer[64];
    FILE *fp;

    // --- Success case ---
    fp = fmemopen(buffer, sizeof(buffer), "w+");
    if (fp == NULL)
      return -1;

    int ret = fputs_unlocked(msg, fp);
    if (ret < 0)
      return -1;
    assert(ret >= 0); // success: must not be EOF

    fflush(fp);
    rewind(fp);

    char readback[64];
    char *res = fgets(readback, sizeof(readback), fp);
    assert(res != NULL);
    fclose(fp);

    // --- Failure case ---
    // Trying to write after the stream is closed
    ret = fputs_unlocked("test", fp);
    assert(ret == EOF); // should fail

    return 0;
}

