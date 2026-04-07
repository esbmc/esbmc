/* arr_assigns_pass:
 * Function declares __ESBMC_assigns(arr[idx]) — allowed to modify arr[idx].
 * Body correctly writes only arr[idx].
 *
 * Phase 2B: array element assigns compliance.
 * Expected: VERIFICATION SUCCESSFUL
 */
int arr_write(int *arr, int idx, int val)
{
  __ESBMC_requires(arr != (int *)0 && idx >= 0 && idx < 10);
  __ESBMC_assigns(arr[idx]);
  __ESBMC_ensures(__ESBMC_return_value == 0);
  arr[idx] = val;
  return 0;
}

int main() { return 0; }
