# `string` call argument: a reaching test for each side of a length branch

Both sides of a branch guarded by a `string` argument are reached, and the
generator emits a real, reaching Foundry test for each side rather than
producing no test file at all.

Before, `format_sol_value` had no renderer for STRING, so the call was dropped
and only the `WARNING: No Foundry test cases collected` warning was printed
next to a 100% coverage number. STRING is now surfaced as a renderable argument
type (`effective_sol_type`) and the string's length is reconstructed from the
nondet buffer `_ESBMC_rand_str` in the model (`recover_nondet_string_length`),
rendering a Solidity string literal of that length: `setName("")` for the
`<= 3` side and `setName("aaaa")` (length 4 > 3) for the `> 3` side.

Content is filler (`'a'`), which is faithful for a length-based branch such as
`bytes(n).length > 3`; a content-dependent branch is a documented residual,
tolerated by the try/catch wrap.

Requiring `Generated Foundry coverage test with 2 case(s)` also rules out the
pre-fix output, since that line and `WARNING: No Foundry test cases collected`
are mutually exclusive. `testing_tool.py` treats every line after the argument
line as a REQUIRED regex — it has no disallowed-pattern section — so negatives
have to be expressed this way, and prose belongs here rather than in
`test.desc`.
