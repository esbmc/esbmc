long a;

struct b {
   long c;
   char bytes[];
};

int main() {
   __ESBMC_assume(a < ((18446744073709551615UL) - 1 - sizeof(struct b)));
   struct b *d = malloc(sizeof(struct b) + a + 1);
   d->bytes[a] = '\0';
   assert(d->bytes[a] == 0);
}
