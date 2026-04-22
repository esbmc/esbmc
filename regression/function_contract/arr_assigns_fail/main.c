/* arr_assigns_fail:
 * KNOWNBUG: detecting arr[idx+1] as unauthorized when assigns=arr[idx] requires
 * a nondet-witness approach (∀j. j≠idx → arr[j] unchanged). Not yet implemented.
 * The Phase 2C fix correctly suppresses false positives on arr itself, but
 * no positive check for unauthorized array writes is emitted yet.
 * Function declares __ESBMC_assigns(arr[idx]) but also writes arr[idx+1].
 * This violates the assigns clause.
 *
 * Phase 2B: array element assigns compliance.
 * Expected: VERIFICATION FAILED
 */
int arr_write2(int *arr, int idx, int val)
{
  __ESBMC_requires(arr != (int *)0 && idx >= 0 && idx < 9);
  __ESBMC_assigns(arr[idx]);
  __ESBMC_ensures(__ESBMC_return_value == 0);
  arr[idx] = val;
  arr[idx + 1] = val; /* BUG: arr[idx+1] not in assigns */
  return 0;
}

int main() { return 0; }
