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
editing without leaving the editor, for C, C++, Python and Solidity files. It
places each violated property on its line in the Problems panel (read from
ESBMC's SARIF report), shows the verdict in the status bar and the
counterexample steps in an **ESBMC Counterexample** view, and keeps ESBMC's full
output in an output channel. A CodeLens above each function verifies that
function alone. Its command-palette entries are:

- `ESBMC: Verify file` — verify the active file with flags built from the
  `esbmc.*` settings; `ESBMC: Show current flags` prints the command line those
  settings produce, and `ESBMC: Show output` opens the output channel;
- `ESBMC: Verify file with Local AI` — verify, then ask a local
  [Ollama](https://ollama.com) model (by default `llama3.1:8b`) to explain the
  counterexample and suggest a fix;
- `ESBMC: Install latest version` and `ESBMC: Update to latest version` —
  download the release build for Linux, macOS or Windows into the extension's
  own storage, so you do not need ESBMC on your `PATH` beforehand (an `esbmc`
  already on the `PATH` is used by default);
- `ESBMC: Open example program` — open the bundled example.

Note that it requires VS Code 1.68 or later. The extension also ships an MCP
server with a `verify` tool, so AI coding agents can run ESBMC through it; the
repository README shows the configuration.

The extension is not on the VS Code Marketplace yet, so build the `.vsix` from
source and install it from the Extensions view; the repository README has the
walkthrough. Sideloading this way does not pull in the extension's dependency
on `mindaro-dev.file-downloader`, which the install and update commands need.

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
