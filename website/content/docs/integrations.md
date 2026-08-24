---
title: Editor and Tool Integration
weight: 16
---

Besides the command line, ESBMC has three interactive front-ends: an editor
extension, a browser GUI, and a plugin for an AI coding agent. All three drive
the same `esbmc` binary, so everything under [Usage](/docs/usage) applies to
them too. For verification in CI rather than at the keyboard, see
[GitHub Action](/docs/github-action).

{{< cards >}}
{{< card link="https://github.com/esbmc/vscode-esbmc" title="VS Code Extension" icon="external-link" >}}
{{< card link="https://github.com/esbmc/esbmc-web" title="Web Interface" icon="external-link" >}}
{{< card link="https://github.com/esbmc/agent-marketplace" title="Claude Code Plugin" icon="external-link" >}}
{{< /cards >}}

## Visual Studio Code

[vscode-esbmc](https://github.com/esbmc/vscode-esbmc) verifies the file you are
editing without leaving the editor. It contributes four command-palette entries:

- `ESBMC: Verify file` — run ESBMC on the active C, C++, Python, Solidity or
  Jimple file and stream the output to the integrated terminal;
- `ESBMC: Verify file with Local AI` — run ESBMC on the current file and, when
  a property fails, ask a local [Ollama](https://ollama.com) model to explain
  the counterexample in a separate output channel. It ignores the `esbmc.*`
  settings and needs Ollama serving `llama3.1:8b`;
- `ESBMC: Install latest version` and `ESBMC: Update to latest version` —
  download and unpack the latest release into `$HOME/bin`, so you do not need
  ESBMC on your `PATH` beforehand.

Note that it requires VS Code 1.68 or later. The install and update commands are Linux
only; elsewhere, install ESBMC first via [Setup](/docs/setup).

The extension is not on the VS Code Marketplace yet, so build the `.vsix` from
source and install it from the Extensions view; the repository README has the
walkthrough. Sideloading this way does not pull in the extension's dependency
on `mindaro-dev.file-downloader`, which the install and update commands need.
Publishing to the Marketplace and Open VSX is tracked in
[vscode-esbmc#15](https://github.com/esbmc/vscode-esbmc/issues/15).

## Web interface

[ESBMC-Web](https://github.com/esbmc/esbmc-web) is a browser GUI for C, C++ and
Python. You write or upload a file, plus any dependency headers or modules,
pick the checks and solver from a form rather than remembering flag names, and
read the result either as the raw ESBMC log or as a dashboard that tabulates
each violation with its file, function and line and shows the counterexample.

It is self-hosted rather than a public service. A Flask backend shells out to
your local `esbmc` and exposes its API on `http://127.0.0.1:5000`; the frontend
is a static page you open from disk at `frontend/index.html`. The repository
documents a WSL path for Windows and a manual install for Linux and macOS.

## Claude Code

The [ESBMC plugin](https://github.com/esbmc/agent-marketplace) brings ESBMC into
[Claude Code](https://docs.anthropic.com/claude-code) for C, C++, Python,
Solidity and Java/Kotlin. It provides a `/verify` command for a quick check of a
source file, an `/audit` command that runs several verification passes for a
security review, and a skill that triggers automatically when a conversation
turns to verifying code — together with reference documentation, examples and
utility scripts.
