# foundry_t20_forge_assert_types_pass

Locks conformance of forge-std `assertEq` type overloads that the generic
name-intercept path already handles (no dedicated code): bool, address, bytes32,
and string (content equality). `test_types_ok` asserts one equal pair of each →
SUCCESSFUL. Verified against real forge 1.7.1 + forge-std v1.16.2 (all PASS).
