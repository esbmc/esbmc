typedef unsigned int size_t;

void *malloc(size_t size);

typedef struct {
  enum blockstate bs;
  int id;
  int version;
} block;

typedef struct blocknode {
  block* b;
  struct blocknode* next;
} blocknode;

typedef blocknode* bl;

int main () {
  block* bp = (block*)(malloc(sizeof(block)));
  __ESBMC_assume(bp);
  bl l = (bl)(malloc(sizeof(blocknode)));
  __ESBMC_assume(l);
  bp->version = 1;
  l->b = bp;
  l->b->version = l->b->version+1;
  assert ((l->b->version) == 1);
}
