#include <string.h>
#include <assert.h>

typedef struct
{
  const char *owner_;
} BankAccount;

BankAccount init(const char *owner)
{
  BankAccount ba;
  ba.owner_ = owner;
  return ba;
}

int main(void)
{
  BankAccount b = init("Alice");
  assert(strcmp("Alice", b.owner_) == 0);
  return 0;
}
