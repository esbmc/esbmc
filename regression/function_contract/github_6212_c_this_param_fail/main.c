/* github_6212_c_this_param_fail:
 * The receiver exemption is keyed on the C++ implicit receiver. `this` is a
 * reserved word only in C++; in C it names a parameter like any other and
 * carries no one-object guarantee, so it must get the same nondet extent as
 * any other struct pointer parameter.
 */
typedef struct
{
  int x;
} S;

void f(S *this)
{
  __ESBMC_requires(this != 0);
  __ESBMC_ensures(1);
  this->x = 1;
}

int main()
{
  return 0;
}
