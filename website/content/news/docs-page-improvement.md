---
title: "Docs Page Improvement"
date: 2026-06-06T20:26:47+01:00
author: Yiannis Charalambous
draft: false
tags:
  - ESBMC
  - Documentation
  - OpenSource
---

📚 We have given the ESBMC documentation a thorough cleanup.

When the docs moved to this site, much of the content was carried over verbatim
from the old wiki: one enormous documentation page that mixed every topic
together, duplicated other pages, and was hard to navigate. We have reorganized
it into something far easier to read.

**Build guide, by platform.** The [build guide](/docs/development/building) is
now a set of per-OS tabs — Ubuntu/Debian, Fedora, macOS, FreeBSD and Windows —
where each tab is a self-contained sequence of steps with the optional bits
tucked into collapsible sections. The duplicate Linux and Windows build pages
that used to confuse contributors are gone.

**Focused pages instead of one wall of text.** The 1000-plus-line documentation
landing page has been split into focused topics:

- [Modeling with non-determinism](/docs/theory/non-determinism) and the
  [verification algorithms](/docs/theory/verification-algorithms) now live under
  Theory.
- [Loop invariants](/docs/loop-invariants) get their own page, and the
  [quantifier constructs](/docs/constructs) join the rest of the verification
  constructs.
- Verification strategies, multi-property verification and the SMT-backend
  reference moved into the [Usage](/docs/usage) guide.

**A real index.** The [documentation home](/docs) is now a concise directory
that points you to each area and invites you into the background theory, instead
of being a page you have to scroll through.

The result is smaller, better organized, and much more readable.

[Go check it out!](/docs)
