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

Create a tag for the commit with the version change:

```sh
git tag v8.0
```

Push the tag to origin:

```sh
git push --tags
```

### Creating the Release on GitHub

After pushing the tag, GitHub will allow you to create the official release
associated with that tag through the repository’s Releases page.

{{% /steps %}}
