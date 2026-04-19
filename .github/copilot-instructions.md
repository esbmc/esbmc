# GitHub Copilot Review Instructions for ESBMC

## Project Context
ESBMC is a context-bounded model checker for embedded C/C++ software. Reviews should prioritize correctness, safety, and maintainability given the critical nature of formal verification tools.

## Key Review Focus Areas

### 1. Correctness & Safety
- **Memory safety**: Check for potential memory leaks, buffer overflows, use-after-free, and null pointer dereferences
- **Resource management**: Verify proper RAII patterns, smart pointer usage, and resource cleanup
- **Undefined behavior**: Flag any code that might exhibit undefined behavior in C/C++
- **Logic errors**: Verify algorithmic correctness, especially in SMT encoding and verification logic

### 2. Code Quality
- **API consistency**: Ensure new APIs follow existing patterns in the codebase
- **Error handling**: Check for proper error propagation and informative error messages
- **Code duplication**: Identify opportunities to refactor repeated logic
- **Complexity**: Flag overly complex functions that should be broken down

### 3. Performance
- **Algorithm efficiency**: Review computational complexity of verification algorithms
- **Memory usage**: Check for unnecessary allocations or copies, especially in hot paths
- **SMT solver interactions**: Ensure efficient solver query construction and usage

### 4. Testing & Documentation
- **Test coverage**: Verify that new features include appropriate test cases
- **Regression tests**: Ensure changes don't break existing verification capabilities
- **Documentation**: Check that complex logic is properly commented
- **API documentation**: Verify public interfaces are documented

### 5. Verification-Specific Concerns
- **Soundness**: Ensure changes don't compromise verification soundness
- **Completeness**: Check that error detection capabilities aren't reduced
- **Symbolic execution**: Review changes to symbolic state handling
- **SMT encoding**: Verify correctness of constraint generation

### 6. Build & Compatibility
- **Cross-platform compatibility**: Check for platform-specific code that might break builds
- **Dependencies**: Flag introduction of new dependencies without justification
- **Build system**: Verify CMake changes are correct and portable

## Review Priorities
1. **Critical**: Soundness issues, memory safety bugs, undefined behavior
2. **High**: Logic errors, performance regressions, missing tests
3. **Medium**: Code quality, documentation gaps, minor inefficiencies
4. **Low**: Style preferences, minor refactoring opportunities

## Style Guidelines
- Follow existing code style in the file being modified
- Prefer modern C++ idioms (C++11 and later features)
- Use const-correctness throughout
- Prefer stack allocation over heap when possible

## What to Skip
- Don't flag minor style inconsistencies if they match the surrounding code
- Don't suggest refactoring stable, working code unless there's a clear benefit
- Don't request documentation for obviously self-explanatory code
