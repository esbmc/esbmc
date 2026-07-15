---
title: Releases
---

This is a guide on preparing and releasing new ESBMC versions.

> [!WARNING] Warning
>
> This guide is meant for developers of ESBMC with permissions for releasing new
> versions. It will not work if you don't have appropriate permissions.

To create a new ESBMC release on GitHub, follow these steps:

{{% steps %}}

### Create a Branch for the Release

Create a branch appropriately named for this release and switch to it, in this
case:

```sh
git switch -c v8.1
```

### Update the Release Notes

Update the release notes in `/scripts/release-notes.txt` to reflect the changes
included in the new version.

> [!TIP] Tip
>
> You can use Claude or any other LLM for this, just use this command to get the
> differences from the last tag
> `git log v8.0..master --oneline > /tmp/commits-for-release.txt` (in this case
> being `v8.0`) and instruct it to update the file for the next version.

### Update the Project Version

Update the project version by modifying `/CMakeLists.txt`. Specifically the
macros `ESBMC_VERSION_MAJOR`, `ESBMC_VERSION_MINOR`, `ESBMC_VERSION_PATCH` to
include the desired version number.

### Commit and Push to the Branch

Commit the changes and push them to the branch created earlier. Then submit a
PR. Name the commit appropriately, in this case `update: bump version to v8.1`.

```sh
git add scripts/release-notes.txt CMakeLists.txt
git commit -m "update: bump version to v8.1"
git push
```

### Create the PR

> [!NOTE]
>
> You can do this from the GitHub UI as well. But for convenience I show the
> commands.

Make sure to change the title and body to your version number.

```sh
gh pr create \
    --title "update: bump version to v8.1" \
    --body "Bump version to v8.1 and update release notes."
```

### Await PR to be Merged

Allow 2 other maintainers to review your PR as usual, approve and merge it.

### Tagging the Commit

Create a tag for the commit with the version change, make sure to name it
according to the version you're releasing. In this case `v8.1`:

```sh
git tag v8.1
```

Push the tag to origin:

```sh
git push --tags
```

> [!IMPORTANT]
>
> Pushing a `v*` tag automatically triggers the
> [Release](https://github.com/esbmc/esbmc/actions/workflows/release.yml)
> workflow ("Upload Release Asset"). It builds every platform and creates a
> **draft** GitHub release named after the tag (e.g. `ESBMC v8.1`), with the
> canonical `esbmc-linux.zip`, `esbmc-windows.zip`, `esbmc-macos.zip` and
> `esbmc.info` assets already attached. You do **not** need to run the workflow
> manually, create the release by hand, or upload any artifacts yourself.

### Publish the Draft Release

Wait for the Release workflow to finish, then open the repository’s
[Releases](https://github.com/esbmc/esbmc/releases) page and find the draft
release the workflow created for your tag.

1. Confirm the attached assets are present: `esbmc-linux.zip`,
   `esbmc-windows.zip`, `esbmc-macos.zip` and `esbmc.info`.
2. Adjust the title if needed — the draft is created as `ESBMC v8.1`; past
   releases use "Release 8.1" (note the missing v).
3. Replace the placeholder description with a markdown version of the content
   written in `/scripts/release-notes.txt` from earlier.
4. Publish the release.

> [!TIP]
>
> You can use Claude or any other LLM to prepare the release description.

{{% /steps %}}
