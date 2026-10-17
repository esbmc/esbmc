# foundry_t20_forge_assert_bytes_pass

`assertEq(bytes, bytes)` lowered to the PRECISE content comparison
`bytes_dynamic_equal(a, b, $dynamic_pool)` (reusing the C-model helper). Two
content-equal dynamic-bytes values → SUCCESSFUL. Matches real forge (PASS).
Guards against the reference-equality regression that would report equal-content
bytes as FAILED (a false WRONG).
