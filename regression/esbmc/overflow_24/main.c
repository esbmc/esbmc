// standard h files
#include <stdbool.h>
#include <stdint.h>

typedef struct struct1_s
{
  bool field0;
  uint32_t field1;
  uint32_t field2;
  uint32_t field3;
  uint64_t field4;
} struct1_t;

typedef union
{
  uint32_t raw32;

  struct
  {
    uint32_t field0 : 2;
    uint32_t field1 : 30;
  };
} struct2_t;

extern struct2_t global_extern_var;

uint64_t bar(bool bool_param1, bool bool_param2)
{
  int16_t local_var1 = nondet_int();
  int32_t local_var2 = nondet_int() << 1;

  local_var1 = local_var1 + local_var2;

  return local_var1;
}

void foo(struct1_t *struct1_var_ptr)
{
  bool bool_param1 = struct1_var_ptr->field0;
  uint64_t local_var1 = bar(bool_param1, 0);
}

int main(void)
{
  struct1_t struct1_var;
  foo(&struct1_var);
  return 0;
}
