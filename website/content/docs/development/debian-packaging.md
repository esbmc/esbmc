---
title: Debian Packaging Guide
---

This document explains how ESBMC is packaged for the **official Debian archive**
and how to maintain that package. It is aimed at anyone who wants to update the
package for a new upstream release or take over its maintenance.

> [!NOTE]
>
> This is **distinct** from the Ubuntu PPA. The PPA
> ([PPA Maintenance](/docs/development/ppa-maintenance)) ships pre-built
> binaries to Launchpad for Ubuntu users. The Debian package goes through the
> Debian archive's review process (mentors + a Debian Developer sponsor) and is
> built from source on Debian `unstable` (sid).

## Overview

- Debian source package name: `esbmc`
- Packaging is developed on Salsa: <https://salsa.debian.org/WeiqiWang/esbmc>
- Uploads for sponsorship go to <https://mentors.debian.net/package/esbmc/>
- Sponsorship is tracked by an **RFS** ("Request For Sponsorship") bug against
  the `sponsorship-requests` pseudo-package.

Debian does not let a non–Debian-Developer upload directly. The workflow is:
build a correct source package → upload it to mentors → file an RFS bug → a
Debian Developer reviews it and, if satisfied, uploads it on your behalf.

## The Five Essentials

Modern Debian packaging converges on five conventions. Every choice below
follows them:

1. **`dh`** — the `debhelper` sequencer drives `debian/rules`.
2. **`3.0 (quilt)`** — the source package format; upstream stays pristine and
   all Debian changes live as patches under `debian/patches/`.
3. **git** — the packaging history is kept in git.
4. **Salsa** — Debian's GitLab hosts the packaging repository and runs CI.
5. **DEP-5** — the machine-readable `debian/copyright` format.

## Before You Begin

### Install the toolchain

```bash
sudo apt install build-essential devscripts debmake debhelper \
                 dh-make quilt git-buildpackage \
                 lintian sbuild mmdebstrap piuparts
```

### Set your identity

`devscripts` tools read your name and email from the environment. Put these in
`~/.bashrc` so every tool (and the changelog) is consistent:

```bash
export DEBEMAIL="lukewang19@icloud.com"
export DEBFULLNAME="Weiqi Wang"
```

The **same** identity — email, OpenPGP key, and Salsa SSH key — must match
across `debian/changelog`, the GPG signature on the upload, and your Salsa
account. Mismatches are the most common reason an upload or a push is rejected.

### Read the manual first

Debian packaging practice changes over time, and the authoritative sources are
online — not your local tooling and not an LLM's training data. Before touching
`debian/`, read at least:

- [Debian Debmake Manual](https://www.debian.org/doc/manuals/debmake-doc/) —
  the modern hands-on tutorial (Chapter 5, "Simple packaging", is the core).
- [Debian Policy Manual](https://www.debian.org/doc/debian-policy/) — the
  normative rules.
- [Debian Developer's Reference](https://www.debian.org/doc/manuals/developers-reference/)
  — process and etiquette.

> [!TIP] Tip
>
> The single source of truth for *current* practice is a clean `sid` build plus
> the online Policy/lintian. An old host (e.g. an LTS release) ships an old
> `lintian`/`debhelper` that will not warn about today's issues.

## Where Things Live

- **Salsa packaging repo**: `git@salsa.debian.org:WeiqiWang/esbmc.git`
  - `debian/latest` — the packaging branch (default).
  - `upstream/latest` — imported upstream sources.
  - Tag `upstream/8.4.0+dfsg` — the DFSG-clean upstream import.
  - Layout follows `git-buildpackage`; `debian/gbp.conf` sets `pristine-tar =
    False`.
- **CI**: `debian/salsa-ci.yml` runs a full `sid` build and the standard checks
  (this satisfies the "build and test on sid" requirement even without a local
  sid box).

## Version Scheme

ESBMC ships a `+dfsg` repack (see [Producing a DFSG-clean
tarball](#producing-a-dfsg-clean-tarball) below), so the version looks like:

```
8.4.0+dfsg-1
└───┘ └──┘ └┘
  │     │   └ Debian revision (starts at 1; bump for packaging-only changes)
  │     └──── +dfsg marks a repacked upstream tarball
  └────────── upstream version (matches the v8.4 git tag)
```

Until the package is accepted into Debian, keep the revision at `-1` and
overwrite the upload on mentors rather than bumping — there is no history to
preserve yet.

## Packaging Workflow

### Producing a DFSG-clean tarball

Debian only ships source that complies with the Debian Free Software
Guidelines. ESBMC's upstream tree contains a few non-source or non-free bits
(pre-built Qt binaries, `.DS_Store`, IDE `*.pro.user` files) that must be
stripped. `debian/copyright` lists them under `Files-Excluded:`, and
`mk-origtargz` applies that list while repacking:

```bash
# Export the exact upstream tag, then repack it into a +dfsg orig tarball
git archive --format=tar --prefix=esbmc-8.4.0/ v8.4 | xz > /tmp/esbmc-8.4.0.tar.xz
mk-origtargz --repack --compression xz --repack-suffix +dfsg \
             --copyright-file debian/copyright \
             /tmp/esbmc-8.4.0.tar.xz
```

This produces `esbmc_8.4.0+dfsg.orig.tar.xz` with the excluded files removed.

> [!IMPORTANT]
>
> When (re)building the Salsa git repo from scratch, upstream's `.gitignore`
> contains `build*` (no leading slash), which matches at **every** level and
> silently drops tracked source files on a fresh `git add -A`. Force past it
> with `git add -f -A .` and verify the staged file count matches the tarball's.

### debian/control — declare minimal, justified dependencies

List only what the build actually uses, and be ready to justify every line. The
best evidence is the upstream build system itself.

**Worked example — Boost.** ESBMC's `CMakeLists.txt` asks for exactly these
components:

```cmake
find_package(Boost REQUIRED
  COMPONENTS date_time program_options iostreams
  OPTIONAL_COMPONENTS system filesystem)
```

So the Build-Depends needs those five component `-dev` packages plus the core
headers (many parts of ESBMC use header-only Boost via `${Boost_INCLUDE_DIRS}`)
— **not** the `libboost-all-dev` metapackage, which pulls in dozens of unused
libraries:

```
libboost-dev,
libboost-date-time-dev,
libboost-program-options-dev,
libboost-iostreams-dev,
libboost-filesystem-dev,
libboost-system-dev,
```

> [!TIP] Tip
>
> Only pin a version (`foo (>= X)`) when you *know* a feature needs it, and add
> a short comment saying why. Runtime library dependencies are computed
> automatically by `${shlibs:Depends}` — do not hand-write shared-library
> versions. Do not list packages that `build-essential` already provides (e.g.
> `libc6-dev`); `lintian` flags that as an error.

### debian/rules — keep it to `dh`

A CMake project needs almost nothing beyond the sequencer:

```makefile
#!/usr/bin/make -f
%:
	dh $@
```

Add an `override_dh_auto_configure` only if you must pass CMake flags. With
`debhelper-compat (= 14)`, `dh` exports `SOURCE_DATE_EPOCH` and handles
reproducible-build concerns for you — never embed the build machine's wall-clock
time. (ESBMC's version string uses the git hash, not a build timestamp, so there
is nothing extra to do here.)

### Building in a clean sid chroot with sbuild

A clean chroot contains **only** your declared Build-Depends, so it is the only
way to prove the dependency list is both sufficient and minimal — if you trimmed
too much, the build fails immediately.

```bash
# One-time: create a sid chroot (rootless unshare backend)
mmdebstrap --variant=buildd --components=main \
           unstable ~/.cache/sbuild/unstable-amd64.tar.zst

# Build the source package, then build it in the chroot
dpkg-buildpackage -S -sa -us -uc
sbuild --dist=unstable ../esbmc_8.4.0+dfsg-1.dsc
```

> [!TIP] Tip
>
> On a non-Debian host (e.g. Ubuntu) `mmdebstrap` may fail with `NO_PUBKEY`
> because the host lacks current Debian archive keys. Install a recent
> `debian-archive-keyring` and point `mmdebstrap` at it, or drop its
> `trusted.gpg.d/*.asc` files into `/etc/apt/trusted.gpg.d/`. If your host's
> `sbuild` is too old to drive the unshare backend, rely on Salsa CI for the
> authoritative sid build.

### Running lintian

```bash
lintian --profile debian --info ../esbmc_8.4.0+dfsg-1_source.changes
```

Fix every error and warning. Understand — do not just silence — each tag; the
`--info` flag prints an explanation and a policy reference for each one.

### Signing and uploading to mentors

```bash
debsign -k <YOUR_GPG_FINGERPRINT> ../esbmc_8.4.0+dfsg-1_source.changes
dput mentors ../esbmc_8.4.0+dfsg-1_source.changes
```

If you re-upload the same version, `dput` skips it because its `.upload` log
already lists the file — force it with `dput -f mentors ...`.

### Filing the RFS and working with a sponsor

File a Request-For-Sponsorship bug so a Debian Developer knows the package is
ready for review:

```bash
# reportbug guides you through it; the bug goes to sponsorship-requests
reportbug --severity=wishlist sponsorship-requests
```

Then wait for a sponsor. When they review, expect questions about specific
choices ("why this dependency?", "why this version?"). Answer them from your own
understanding — that, not the files, is what earns the upload.

## After Acceptance: Ongoing Maintenance

- **New upstream release**: `uscan` (driven by `debian/watch`) downloads and
  repacks the new tarball; import it with `gbp import-orig --uscan`, update
  `debian/changelog` with `dch`, rebuild, re-test on sid, and upload.
- **Bugs**: handle reports via the Debian BTS (`bts` from `devscripts`). Treat
  each report as useful signal, not criticism.
- **Standards**: bump `Standards-Version` when Policy changes and re-check with
  `lintian`.

## Working with AI Assistance

Using an LLM to draft packaging is acceptable, but the maintainer — not the
tool — is accountable for the result. The practical rules are the same ones
Homebrew states explicitly in its
[Responsible AI Usage](https://docs.brew.sh/Responsible-AI-Usage) policy and
that Debian reviewers expect informally:

- Read, run, and test every change yourself before asking a human to review it.
- Be able to explain and defend **every** field in `debian/` in your own words.
- LLM training data goes stale — verify current practice against a clean `sid`
  build and the online Policy/lintian, not the model's memory.
- Be able to address every review comment yourself, even when the tool cannot.

## Common Issues

### `debhelper-compat` mismatch on an old host

**Symptom**: `dpkg-buildpackage` refuses because the host's `debhelper` is older
than the compat level in `debian/control`.

**Fix**: build the source package with `-d` (skip the build-dep check) and do
the real build in an sbuild sid chroot, which has the correct `debhelper`.

### mentors: "Package has already been uploaded"

**Fix**: `dput -f mentors ../esbmc_*_source.changes` to force a re-upload of the
same version.

### lintian error: `build-depends-on-build-essential-package-without-using-version`

**Cause**: listing a package that `build-essential` already guarantees (e.g.
`libc6-dev`) without a version.

**Fix**: remove the dependency entirely — it is implied.

## Useful Commands

```bash
# Inspect a built source package
dpkg-source -I ../esbmc_*.dsc

# Compare two builds of the package
debdiff ../esbmc_8.4.0+dfsg-1.dsc ../esbmc_8.4.0+dfsg-2.dsc

# Check the watch file resolves the latest upstream
uscan --report --verbose

# Verify version ordering
dpkg --compare-versions 8.4.0+dfsg-1 lt 8.4.0+dfsg-2 && echo ok
```

## Related Resources

- [Debian Debmake Manual](https://www.debian.org/doc/manuals/debmake-doc/)
- [Debian Policy Manual](https://www.debian.org/doc/debian-policy/)
- [Debian Developer's Reference](https://www.debian.org/doc/manuals/developers-reference/)
- [Debian Mentors](https://mentors.debian.net/)
- [Debian Code Search](https://codesearch.debian.net/) — find how other packages solve a problem
- [ESBMC PPA Maintenance Guide](/docs/development/ppa-maintenance)

---

**Maintainer**: Weiqi Wang <lukewang19@icloud.com>
**Last Updated**: 2026-07-24
