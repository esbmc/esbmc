#define NULL 0

void *malloc(unsigned size);

struct list_el {
  int val;
  struct list_el *next;
  int x;
};

typedef struct list_el item;

int main()
{
  item *curr, *head;

  head = NULL;

  curr = (item *)malloc(sizeof(item));
  __ESBMC_assume(curr);
  curr->val = 1;
  curr->x = 2;
  curr->next  = head;
  head = curr;

  curr = (item *)malloc(sizeof(item));
  __ESBMC_assume(curr);
  curr->val = 2;
  curr->x = 3;
  curr->next  = head;
  head = curr;

  assert(head->val==2);
  assert(head->x==3);

}
