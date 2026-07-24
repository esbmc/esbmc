---
title: Homebrew Packaging Guide
---

This document explains how ESBMC can be packaged for
[Homebrew](https://brew.sh), the macOS/Linux package manager, and what a
maintainer must do to get a formula accepted and keep it current.

> [!NOTE]
>
> Status: ESBMC is **not yet in `homebrew/core`**. This is a how-to and a
> reading list for adding and maintaining a formula, not a description of an
> existing one.

## Overview

Homebrew has three kinds of packages:

- **Formula** — open-source software Homebrew builds from source. This is what
  ESBMC would be.
- **Cask** — a pre-built binary application (not applicable here).
- **Tap** — a third-party repository of formulae. You can ship a formula from
  your own tap immediately; getting it into the official `homebrew/core` tap
  requires meeting the acceptance policy below.

## Acceptance Requirements for `homebrew/core`

Before writing anything, confirm ESBMC qualifies (it does):

- **Open-source with a DFSG-compatible license** — ESBMC is Apache-2.0. ✓
- **A stable, upstream-tagged version** — Homebrew requires a tagged release
  with a downloadable tarball (git checkouts and untagged software are rejected
  because bottles cannot be built for them). ESBMC tags releases as `vX.Y`
  (e.g. `v8.4`). ✓
- **Buildable from source** (or a cross-platform binary). ✓
- **Notability** — the project must be established and widely used; new formulae
  are held to a higher standard than existing ones.

> [!IMPORTANT]
>
> Check `homebrew/core` for prior pull requests first — a previous rejection may
> record an unresolved licensing, security, or maintenance concern.

## Manuals to Read First

Homebrew's rules are documented; read these before submitting:

- [Formula Cookbook](https://docs.brew.sh/Formula-Cookbook) — the authoritative
  formula DSL and testing reference.
- [Acceptable Formulae](https://docs.brew.sh/Acceptable-Formulae) — what
  qualifies for `homebrew/core`.
- [Package Acceptance Policy](https://docs.brew.sh/Package-Acceptance-Policy) —
  general standards.
- [How To Open a Homebrew Pull Request](https://docs.brew.sh/How-To-Open-a-Homebrew-Pull-Request).
- [Responsible AI Usage](https://docs.brew.sh/Responsible-AI-Usage) — required
  reading if you used an LLM (see below).
- [Homebrew/homebrew-core Maintainer Guide](https://docs.brew.sh/Homebrew-homebrew-core-Maintainer-Guide).

## Creating the Formula

```bash
# Generate a template from the release tarball
brew create https://github.com/esbmc/esbmc/archive/refs/tags/v8.4.tar.gz
```

Then refine the generated Ruby formula:

- Use upstream's canonical homepage (`https://esbmc.org`) and an immutable
  source URL (the tagged tarball), verified with a SHA-256.
- Declare **only** required dependencies (ESBMC needs a C++ toolchain, CMake,
  Boost, Z3, GMP, fmt, yaml-cpp, and LLVM/Clang — mirror the real build system,
  as in the [Debian dependency example](/docs/development/debian-packaging#debiancontrol--declare-minimal-justified-dependencies)).
- Add a meaningful `test do` block — for a verifier, feed ESBMC a tiny program
  with a known bug and assert it reports `VERIFICATION FAILED`, plus a safe
  program that reports `VERIFICATION SUCCESSFUL`.

## Building, Testing, and Auditing

Run the same checks Homebrew CI runs, locally, before submitting:

```bash
# Build from source exactly as a reviewer would
HOMEBREW_NO_INSTALL_FROM_SOURCE=1 brew install --build-from-source esbmc

# Style, audit (for a new formula), and the formula's own test block
brew style esbmc
brew audit --new --strict --online esbmc
brew test esbmc

# Or run the combined online checks in one shot
brew lgtm --online
```

## Opening the Pull Request

```bash
# Branch from the latest default branch of your homebrew-core fork
git checkout -b esbmc-new origin/HEAD

# After editing the formula, commit with Homebrew's "<name> <version>" convention
git commit -am "esbmc 8.4.0"

git push --set-upstream <your-fork> esbmc-new
```

Then open the PR on GitHub and explain the rationale.

## Working with AI Assistance

Homebrew's [Responsible AI Usage](https://docs.brew.sh/Responsible-AI-Usage)
policy permits LLM assistance **with disclosure and full human accountability**:

- **Disclose** in the pull request that AI was used, and explain how you
  verified the changes (the PR template asks for this).
- "AI is not responsible for its output: **you** are responsible for the output
  of the AI tools you use."
- **Review it yourself first** — do not ask other humans to review AI-generated
  code until you have read it, run it, tested it, and understood it.
- **Scrutinize mission-critical code** — anything touching installation,
  security, downloads, or the `Formula` DSL — more carefully.
- You must be able to **address every review comment yourself**, even when the
  AI cannot.

## Related Resources

- [Homebrew Documentation](https://docs.brew.sh/)
- [Formula Cookbook](https://docs.brew.sh/Formula-Cookbook)
- [Acceptable Formulae](https://docs.brew.sh/Acceptable-Formulae)
- [ESBMC Debian Packaging Guide](/docs/development/debian-packaging)
- [ESBMC PPA Maintenance Guide](/docs/development/ppa-maintenance)

---

**Maintainer**: Weiqi Wang <lukewang19@icloud.com>
**Last Updated**: 2026-07-24
