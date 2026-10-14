#include <stdatomic.h>
#include <stdbool.h>

int
main(void) {
        atomic_bool x = false;
        (void)atomic_compare_exchange_strong(&x, &(bool){ false }, true);

        return 0;
}
