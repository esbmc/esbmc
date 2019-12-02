#include <build/btorconfig.h>
#include <src/boolector.h>

int
main() {
    Btor *face = boolector_new();
    (void) face;
    printf("%s", BTOR_VERSION);
    return 0;
}