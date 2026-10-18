# `bytes` call argument: both sides of a length branch get a reaching test

Both sides of `d.length > 3` are reached, and each now gets its own reaching
test case, instead of a single `setData(hex"")` that only exercised the `else`
side while the `> 3` side was silently claimed covered with no test.

`bytes` is renderable by `format_sol_value`: the recovered value is a
`BytesDynamic` struct whose `.length` member is read from the model, and a
zero-filled literal of that length is emitted. The `<= 3` side recovers length
0 → `setData(hex"")`; the `> 3` side recovers an unconstrained-nondet length
(`llc_nondet_bytes` leaves it free, so the solver may pick e.g. 2^64-4), which
is clamped to a small representative (32) that still exceeds the threshold →
`setData(hex"00..00")`, length 32 > 3. The two distinct literals no longer
deduplicate, giving two reaching cases.

Only `.length` is faithfully reconstructed — the byte content lives in a
separate pool — so a content-dependent branch is a documented residual,
tolerated by the try/catch wrap.

Requiring `Generated Foundry coverage test with 2 case(s)` also rules out the
pre-fix outputs: the one-case collapse (`with 1 case(s)`) and the no-test
`WARNING: No Foundry test cases collected`, since those lines are mutually
exclusive with it. `testing_tool.py` treats every line after the argument line
as a REQUIRED regex — it has no disallowed-pattern section — so negatives have
to be expressed this way, and prose belongs here rather than in `test.desc`.
