# foundry_t20_forge_assert_types_fail

Negative dual: `assertEq(bytes32(7), bytes32(8))` → FAILED, confirming the
type-specific equality is genuinely checked (not vacuously passing).
Verified against real forge: `[FAIL] test_b32_bad`.
