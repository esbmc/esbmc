/*
 * Pin contract: smt_conv lowers abs2t as `(x >= 0) ? x : -x`, not the
 * opposite-sense `(x < 0) ? -x : x`. Both shapes are logically
 * equivalent, but bitwuzla's preprocessor produces faster solver time
 * on the `>= 0` form because the surrounding C tends to express sign
 * tests in the non-negative form, so emitting abs in the same direction
 * maximises term-graph sharing in the solver's input. Switching to the
 * `< 0` form pushed sv-benchmarks/c/xcsp/AllInterval-017 from ~7s to
 * timeout (>100s) under bitwuzla.
 *
 * This test exercises the all-interval constraint pattern that triggers
 * the slowdown — 17 nondet vars with all-distinct + abs-difference
 * chain — and pins that the abs lowering completes in well under 60s.
 */

int nondet_int();
extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *)
    __attribute__((__nothrow__, __leaf__)) __attribute__((__noreturn__));
void reach_error() { __assert_fail("0", "abs.c", 0, "reach_error"); }
void assume(int cond) { if (!cond) abort(); }

int main() {
  int v0 = nondet_int(); assume(v0 >= 0); assume(v0 <= 16);
  int v1 = nondet_int(); assume(v1 >= 0); assume(v1 <= 16);
  int v2 = nondet_int(); assume(v2 >= 0); assume(v2 <= 16);
  int v3 = nondet_int(); assume(v3 >= 0); assume(v3 <= 16);
  int v4 = nondet_int(); assume(v4 >= 0); assume(v4 <= 16);
  int v5 = nondet_int(); assume(v5 >= 0); assume(v5 <= 16);
  int v6 = nondet_int(); assume(v6 >= 0); assume(v6 <= 16);
  int v7 = nondet_int(); assume(v7 >= 0); assume(v7 <= 16);
  int v8 = nondet_int(); assume(v8 >= 0); assume(v8 <= 16);
  int v9 = nondet_int(); assume(v9 >= 0); assume(v9 <= 16);
  int v10 = nondet_int(); assume(v10 >= 0); assume(v10 <= 16);
  int v11 = nondet_int(); assume(v11 >= 0); assume(v11 <= 16);
  int v12 = nondet_int(); assume(v12 >= 0); assume(v12 <= 16);
  int v13 = nondet_int(); assume(v13 >= 0); assume(v13 <= 16);
  int v14 = nondet_int(); assume(v14 >= 0); assume(v14 <= 16);
  int v15 = nondet_int(); assume(v15 >= 0); assume(v15 <= 16);
  int v16 = nondet_int(); assume(v16 >= 0); assume(v16 <= 16);

  int v17 = nondet_int(); assume(v17 >= 1); assume(v17 <= 16);
  int v18 = nondet_int(); assume(v18 >= 1); assume(v18 <= 16);
  int v19 = nondet_int(); assume(v19 >= 1); assume(v19 <= 16);
  int v20 = nondet_int(); assume(v20 >= 1); assume(v20 <= 16);
  int v21 = nondet_int(); assume(v21 >= 1); assume(v21 <= 16);
  int v22 = nondet_int(); assume(v22 >= 1); assume(v22 <= 16);
  int v23 = nondet_int(); assume(v23 >= 1); assume(v23 <= 16);
  int v24 = nondet_int(); assume(v24 >= 1); assume(v24 <= 16);
  int v25 = nondet_int(); assume(v25 >= 1); assume(v25 <= 16);
  int v26 = nondet_int(); assume(v26 >= 1); assume(v26 <= 16);
  int v27 = nondet_int(); assume(v27 >= 1); assume(v27 <= 16);
  int v28 = nondet_int(); assume(v28 >= 1); assume(v28 <= 16);
  int v29 = nondet_int(); assume(v29 >= 1); assume(v29 <= 16);
  int v30 = nondet_int(); assume(v30 >= 1); assume(v30 <= 16);
  int v31 = nondet_int(); assume(v31 >= 1); assume(v31 <= 16);
  int v32 = nondet_int(); assume(v32 >= 1); assume(v32 <= 16);

  // All-distinct on v0..v16
  for (int i = 0; i < 17; i++)
    for (int j = i + 1; j < 17; j++)
      ;  // unrolled below
  assume(v0 != v1); assume(v0 != v2); assume(v0 != v3); assume(v0 != v4);
  assume(v0 != v5); assume(v0 != v6); assume(v0 != v7); assume(v0 != v8);
  assume(v0 != v9); assume(v0 != v10); assume(v0 != v11); assume(v0 != v12);
  assume(v0 != v13); assume(v0 != v14); assume(v0 != v15); assume(v0 != v16);
  assume(v1 != v2); assume(v1 != v3); assume(v1 != v4); assume(v1 != v5);
  assume(v1 != v6); assume(v1 != v7); assume(v1 != v8); assume(v1 != v9);
  assume(v1 != v10); assume(v1 != v11); assume(v1 != v12); assume(v1 != v13);
  assume(v1 != v14); assume(v1 != v15); assume(v1 != v16);
  assume(v2 != v3); assume(v2 != v4); assume(v2 != v5); assume(v2 != v6);
  assume(v2 != v7); assume(v2 != v8); assume(v2 != v9); assume(v2 != v10);
  assume(v2 != v11); assume(v2 != v12); assume(v2 != v13); assume(v2 != v14);
  assume(v2 != v15); assume(v2 != v16);
  assume(v3 != v4); assume(v3 != v5); assume(v3 != v6); assume(v3 != v7);
  assume(v3 != v8); assume(v3 != v9); assume(v3 != v10); assume(v3 != v11);
  assume(v3 != v12); assume(v3 != v13); assume(v3 != v14); assume(v3 != v15);
  assume(v3 != v16);
  assume(v4 != v5); assume(v4 != v6); assume(v4 != v7); assume(v4 != v8);
  assume(v4 != v9); assume(v4 != v10); assume(v4 != v11); assume(v4 != v12);
  assume(v4 != v13); assume(v4 != v14); assume(v4 != v15); assume(v4 != v16);
  assume(v5 != v6); assume(v5 != v7); assume(v5 != v8); assume(v5 != v9);
  assume(v5 != v10); assume(v5 != v11); assume(v5 != v12); assume(v5 != v13);
  assume(v5 != v14); assume(v5 != v15); assume(v5 != v16);
  assume(v6 != v7); assume(v6 != v8); assume(v6 != v9); assume(v6 != v10);
  assume(v6 != v11); assume(v6 != v12); assume(v6 != v13); assume(v6 != v14);
  assume(v6 != v15); assume(v6 != v16);
  assume(v7 != v8); assume(v7 != v9); assume(v7 != v10); assume(v7 != v11);
  assume(v7 != v12); assume(v7 != v13); assume(v7 != v14); assume(v7 != v15);
  assume(v7 != v16);
  assume(v8 != v9); assume(v8 != v10); assume(v8 != v11); assume(v8 != v12);
  assume(v8 != v13); assume(v8 != v14); assume(v8 != v15); assume(v8 != v16);
  assume(v9 != v10); assume(v9 != v11); assume(v9 != v12); assume(v9 != v13);
  assume(v9 != v14); assume(v9 != v15); assume(v9 != v16);
  assume(v10 != v11); assume(v10 != v12); assume(v10 != v13); assume(v10 != v14);
  assume(v10 != v15); assume(v10 != v16);
  assume(v11 != v12); assume(v11 != v13); assume(v11 != v14); assume(v11 != v15);
  assume(v11 != v16);
  assume(v12 != v13); assume(v12 != v14); assume(v12 != v15); assume(v12 != v16);
  assume(v13 != v14); assume(v13 != v15); assume(v13 != v16);
  assume(v14 != v15); assume(v14 != v16);
  assume(v15 != v16);

  // |v_i - v_{i+1}| chain
  int var_for_abs;
#define ABS_DIFF(a, b, target) \
  var_for_abs = a - b; \
  var_for_abs = (var_for_abs >= 0) ? var_for_abs : var_for_abs * (-1); \
  assume(target == var_for_abs)

  ABS_DIFF(v0, v1, v17);
  ABS_DIFF(v1, v2, v18);
  ABS_DIFF(v2, v3, v19);
  ABS_DIFF(v3, v4, v20);
  ABS_DIFF(v4, v5, v21);
  ABS_DIFF(v5, v6, v22);
  ABS_DIFF(v6, v7, v23);
  ABS_DIFF(v7, v8, v24);
  ABS_DIFF(v8, v9, v25);
  ABS_DIFF(v9, v10, v26);
  ABS_DIFF(v10, v11, v27);
  ABS_DIFF(v11, v12, v28);
  ABS_DIFF(v12, v13, v29);
  ABS_DIFF(v13, v14, v30);
  ABS_DIFF(v14, v15, v31);
  ABS_DIFF(v15, v16, v32);

  // All-distinct on v17..v32
  assume(v17 != v18); assume(v17 != v19); assume(v17 != v20); assume(v17 != v21);
  assume(v17 != v22); assume(v17 != v23); assume(v17 != v24); assume(v17 != v25);
  assume(v17 != v26); assume(v17 != v27); assume(v17 != v28); assume(v17 != v29);
  assume(v17 != v30); assume(v17 != v31); assume(v17 != v32);
  assume(v18 != v19); assume(v18 != v20); assume(v18 != v21); assume(v18 != v22);
  assume(v18 != v23); assume(v18 != v24); assume(v18 != v25); assume(v18 != v26);
  assume(v18 != v27); assume(v18 != v28); assume(v18 != v29); assume(v18 != v30);
  assume(v18 != v31); assume(v18 != v32);
  assume(v19 != v20); assume(v19 != v21); assume(v19 != v22); assume(v19 != v23);
  assume(v19 != v24); assume(v19 != v25); assume(v19 != v26); assume(v19 != v27);
  assume(v19 != v28); assume(v19 != v29); assume(v19 != v30); assume(v19 != v31);
  assume(v19 != v32);
  assume(v20 != v21); assume(v20 != v22); assume(v20 != v23); assume(v20 != v24);
  assume(v20 != v25); assume(v20 != v26); assume(v20 != v27); assume(v20 != v28);
  assume(v20 != v29); assume(v20 != v30); assume(v20 != v31); assume(v20 != v32);
  assume(v21 != v22); assume(v21 != v23); assume(v21 != v24); assume(v21 != v25);
  assume(v21 != v26); assume(v21 != v27); assume(v21 != v28); assume(v21 != v29);
  assume(v21 != v30); assume(v21 != v31); assume(v21 != v32);
  assume(v22 != v23); assume(v22 != v24); assume(v22 != v25); assume(v22 != v26);
  assume(v22 != v27); assume(v22 != v28); assume(v22 != v29); assume(v22 != v30);
  assume(v22 != v31); assume(v22 != v32);
  assume(v23 != v24); assume(v23 != v25); assume(v23 != v26); assume(v23 != v27);
  assume(v23 != v28); assume(v23 != v29); assume(v23 != v30); assume(v23 != v31);
  assume(v23 != v32);
  assume(v24 != v25); assume(v24 != v26); assume(v24 != v27); assume(v24 != v28);
  assume(v24 != v29); assume(v24 != v30); assume(v24 != v31); assume(v24 != v32);
  assume(v25 != v26); assume(v25 != v27); assume(v25 != v28); assume(v25 != v29);
  assume(v25 != v30); assume(v25 != v31); assume(v25 != v32);
  assume(v26 != v27); assume(v26 != v28); assume(v26 != v29); assume(v26 != v30);
  assume(v26 != v31); assume(v26 != v32);
  assume(v27 != v28); assume(v27 != v29); assume(v27 != v30); assume(v27 != v31);
  assume(v27 != v32);
  assume(v28 != v29); assume(v28 != v30); assume(v28 != v31); assume(v28 != v32);
  assume(v29 != v30); assume(v29 != v31); assume(v29 != v32);
  assume(v30 != v31); assume(v30 != v32);
  assume(v31 != v32);

  reach_error();
  return 0;
}
