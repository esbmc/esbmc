#include <assert.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>

int main(void) {
    int fds[2];
    int ret = pipe(fds);
    assert(ret == 0);

    // Try to seek on the read end of the pipe → should fail
    off_t pos = lseek(fds[0], 0, SEEK_SET);
    assert(pos == -1);
    assert(errno == ESPIPE); // Illegal seek

    close(fds[0]);
    close(fds[1]);
    return 0;
}

