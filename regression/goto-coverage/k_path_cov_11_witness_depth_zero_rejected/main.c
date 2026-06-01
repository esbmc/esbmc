// Regression for #4330 review. The `read_positive` lambda in
// parseoptions guards `--k-path-witness-depth` and `--k-path-max-goals`
// against non-positive values. k_path_cov_6 already covers the equivalent
// check on `--k-path-coverage` itself, but the lambda fires per-flag —
// a regression that broke just the witness-depth path would not be
// caught there. This test exercises that specific guard: setting
// --k-path-witness-depth=0 must abort at parse time, not silently coerce
// the cap to 0 (which would emit zero witnesses and report a vacuous
// "N/A" coverage that could be mistaken for "no branches").
int main()
{
  int a;
  return a > 0 ? 1 : 0;
}
