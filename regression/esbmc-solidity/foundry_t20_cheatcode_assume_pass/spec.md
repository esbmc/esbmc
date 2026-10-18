property: `vm.assume(x < 10)` then assert (fuzz input x is symbolic).
classification: correct-test
forge_truth: PASS
esbmc_expected: CORRECT (SUCCESSFUL)
complexity: {cheatcodes:1(vm.assume, MODELED), assertions:1(native), PUT:yes(1 symbolic uint256), call-depth:0}
note: PUT-lite. _pass asserts x<100 (holds under assume x<10); _fail asserts x<5 (x can be 5..9,
  proving vm.assume prunes EXACTLY x>=10 and does not over-constrain — soundness of the assume model).
