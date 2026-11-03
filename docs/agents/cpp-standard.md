# Consulting the C/C++ Standard

When a C/C++ change concerns standard-defined semantics — undefined behaviour,
implicit conversions, object lifetime, name lookup, overload resolution,
constant evaluation, or similar — consult the relevant standard draft (e.g. the
latest C or C++ working draft on open-std.org, or cppreference for a digestible
summary) before implementing. Cite the section in the commit message or code
comment when it clarifies a non-obvious choice. Skip for routine edits that do
not depend on standard semantics.

**Use the standard and the compiler together — they answer different
questions.** When a change to an operational model under `src/cpp/library/` or
`src/c2goto/library/` adds or moves a version gate (`#if __cplusplus >= …`, a
`constexpr`/`noexcept` qualifier, a conditionally-declared member), consult
both:

- **The standard** says what the rule is, when it changed, and why — the
  semantics, the paper number, the wording worth citing. Read it first; it is
  what goes in the commit message.
- **The compiler and its C++ library** (`clang++` with libc++ here, libstdc++ on
  the Linux CI runners) say what is actually available in a given `-std` mode on
  this target. That is what a user sees when they compile a regression test by
  hand, and what the OM has to reproduce.

They usually agree. Where they do not:

- The implementation offers **more** than the standard requires — libc++ exposes
  `<string_view>` in C++11 — follow the implementation. Code built with that
  toolchain compiles, so rejecting it would be a false `PARSING ERROR` on valid
  input. ESBMC's C++11 `<string_view>` gate is exactly this call (#3387).
- The implementation is **non-conforming** — follow the standard, and leave a
  one-line comment naming the divergence.

Establish the boundary by measurement, never by recall:

1. Write one small probe per behaviour in question.
2. Run each probe through `clang++ -std=<mode> -fsyntax-only` for every mode the
   gate spans — the real library, not the OM.
3. Run the same probes through `esbmc --std <mode>` and diff accept/reject. Any
   cell where they disagree is the defect.
4. When changing an OM header, A/B the header trees directly
   (`clang++ -I <tree>`) — it is seconds, against a full OM rebuild.

Mirror the host library's _shape_, not just its version number: libc++ spells
these `_LIBCPP_CONSTEXPR_SINCE_CXX17`, `_LIBCPP_STD_VER >= N`, and where it
declares a member unconditionally `constexpr` and gates only its callee, do the
same — the accept/reject behaviour and the diagnostic both depend on it. Cite
the paper (e.g. P0426R1) in the commit message; take the boundary from the
implementation.

Pin the `--std` mode in every test that depends on it — see "Pin the mode in
every test" in `docs/agents/regression-tests.md`.
