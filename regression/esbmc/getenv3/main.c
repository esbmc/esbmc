#include <stdlib.h>
#include <string.h>
#include <assert.h>

void test_invalid_name_with_equals(void)
{
  char *result = getenv("VAR=value");
  assert(result == NULL);
}

void test_valid_variable_name(void)
{
  char *result = getenv("PATH");
  if (result != NULL)
  {
    assert(strlen(result) >= 0);
    assert(strlen(result) < 4096);
  }
}

void test_memory_safety(void)
{
  char *result = getenv("HOME");

  if (result != NULL)
  {
    char first_char = result[0];
    size_t len = strlen(result);
    assert(len < 4096);
    assert(result[len] == '\0');
  }
}

void test_string_usage_safety(void)
{
  char *result = getenv("QUERY_STRING");

  if (result != NULL)
  {
    size_t len = strlen(result);
    assert(len >= 0);

    if (len > 0)
    {
      char first = result[0];
      assert(first != '\0');
    }
  }
}

int main(void)
{
  test_invalid_name_with_equals();
  test_valid_variable_name();
  test_memory_safety();
  test_string_usage_safety();
  return 0;
}
