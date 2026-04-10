---
title: Config
weight: 11
---

ESBMC supports specifying options through TOML-formatted config files. To use a config file, export an environment variable:

```sh
export ESBMC_CONFIG_FILE="path/to/config.toml"
```

If no environment file is specified, then the default locations will be checked:

* Windows: `%userprofile%\esbmc.toml`
* UNIX: `~/.config/esbmc.toml`

If nothing is found, then nothing is loaded. If you set the environment variable to the empty string, then it disables the entire config file loading process.

```sh
export ESBMC_CONFIG_FILE=""
```

{{< details title="Example Config File" closed="true" >}}
```toml {filename="esbmc.toml"}
interval-analysis = true
goto-unwind = true
unlimited-goto-unwind = true
k-induction = true
state-hashing = true
add-symex-value-sets = true
k-step = 2
floatbv = true
unlimited-k-steps = false
max-k-step = 100
memory-leak-check = true
context-bound = 2
```
{{< /details >}}

{{< callout type="info" >}}
When submitting a GitHub issue, make sure to include the content of your config
file if you are using one.
{{< /callout >}}