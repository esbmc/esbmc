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
  uint64_t local_var1;
  uint32_t local_var2 = ((global_extern_var.field0) << UINT64_C(0x0000000a));

  if(bool_param1)
    local_var1 = local_var2 + 0xFEB74400;

  return local_var1;
}

void foo(struct1_t *struct1_var_ptr)
{
  bool bool_param1 = struct1_var_ptr->field0;
  uint64_t local_var1 = bar(bool_param1, 0);
}

int main(void)
{
  0 << UINT64_C(0);
  0 << (0ULL);
  struct1_t struct1_var;
  foo(&struct1_var);
  return 0;
}
