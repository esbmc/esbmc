---
title: Updating an SMT Solver
---

This guide explains how to bump the version of an SMT solver already integrated
in ESBMC. The worked example bumps **Bitwuzla**, but the steps generalise — the
**only** thing that changes per solver is *where the version is pinned*, which
depends on how that solver is obtained.

> [!NOTE]
> Anchors below name **files and identifiers** rather than line numbers, which
> drift. Grep for the identifier if you cannot find it.

## Where versions are pinned

ESBMC obtains solvers in two ways, and the pin location differs:

| Solver | How obtained | Version pinned in |
|---|---|---|
| Bitwuzla, Boolector, Yices, CVC4 | built from source (CPM / git) | `GIT_TAG` in `src/solvers/<solver>/CMakeLists.txt` |
| Z3, CVC5, MathSAT | prebuilt archive download | `DEFAULT_<S>_URL` / `DEFAULT_<S>_NAME` in `scripts/cmake/Options.cmake` |

In **both** cases there is also a compatibility floor:
`set(<Solver>_MIN_VERSION "x.y.z")` at the top of
`src/solvers/<solver>/CMakeLists.txt`. ESBMC's CMake aborts the build if the
located library is older than this.

CI workflows and `scripts/build.sh` do **not** pin solver versions — they build
with `-DDOWNLOAD_DEPENDENCIES=On` and inherit whatever the CMake files above
specify. So you usually do not touch `.github/workflows/*` for a version bump.

## Procedure

### A. Source-built solver (e.g. Bitwuzla)

1. **Bump the git tag.** In `src/solvers/bitwuzla/CMakeLists.txt`, update the
   `GIT_TAG` passed to `cpmaddpackage(... GITHUB_REPOSITORY bitwuzla/bitwuzla
   GIT_TAG <new-tag>)`.
2. **Raise the compatibility floor** if the new version is now required:
   `set(Bitwuzla_MIN_VERSION "<new>")`.
3. **Re-check the build recipe.** Source-built solvers run their own configure
   step (Bitwuzla: `python3 ./configure.py … && meson install`). If the upstream
   build system changed flags between versions, update that `execute_process`
   invocation.

### B. Prebuilt-download solver (e.g. Z3, CVC5, MathSAT)

1. **Bump the URL and extracted-directory name** in `scripts/cmake/Options.cmake`
   — update the per-OS `DEFAULT_Z3_URL` / `DEFAULT_Z3_NAME` (and the Windows
   branch) so the new archive is downloaded and found. These feed the cache
   variables `ESBMC_Z3_URL` / `ESBMC_Z3_NAME` consumed by
   `src/solvers/z3/CMakeLists.txt`.
2. **Raise `<Solver>_MIN_VERSION`** in the per-solver `CMakeLists.txt` if needed.

### C. Common to both

3. **Update `BUILDING.md`.** Change the documented version so manual installers
   match what `DOWNLOAD_DEPENDENCIES` fetches.

4. **Reconcile API changes — the step that actually breaks.** If the solver
   changed its API between versions, update the backend interface in
   `src/solvers/<solver>/<solver>_conv.{cpp,h}` (renamed functions, changed
   signatures, new term/sort constructors). The compile-time probe
   `src/solvers/<solver>/try_<solver>.c` may also need adjusting if the symbols it
   references moved.

5. **Watch bundled static dependencies.** Some solvers statically link a SAT core
   that clashes with another backend — e.g. `src/solvers/bitwuzla/CMakeLists.txt`
   renames Bitwuzla's bundled CaDiCaL symbols to avoid colliding with cvc5. A new
   version may bundle a different SAT core or change those symbol names; verify the
   rename block still matches.

## Validate

```sh
# force a fresh fetch/build of the bumped solver
rm -rf build
cmake -GNinja -Bbuild -S . -DDOWNLOAD_DEPENDENCIES=On -DENABLE_BITWUZLA=On \
  -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja -C build
```

- Confirm CMake reports the new version (`message(STATUS "… version: …")`) and
  does not trip the `MIN_VERSION` floor.
- Run that solver's regression subset and compare against the previous version /
  another backend; a new failure usually means an unhandled API or semantic
  change, not a flaky test:
  ```sh
  cd build && ctest -j$(nproc) -L regression --timeout 120
  rm -rf /tmp/esbmc-headers-*   # clean per-test temp dirs
  ```

## Notes & pitfalls

- **Don't forget the `MIN_VERSION` floor.** Bumping `GIT_TAG`/URL without it lets
  an older system-installed library satisfy the build and silently shadow the
  intended version.
- **A version bump can be a behaviour change.** Solver upgrades alter heuristics,
  default options, and occasionally `check-sat` results on hard instances; treat a
  regression diff as signal, and pin/adjust solver options in the backend rather
  than working around it in tests.
- **Keep `BUILDING.md` and the CMake pin in lockstep** — they are the two places a
  user vs. the automated build read the version from, and drift between them
  produces "works in CI, fails locally" reports.
