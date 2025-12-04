# ESBMC PPA Maintenance Guide

This document explains how to maintain ESBMC's Launchpad PPA (Personal Package Archive) for maintainers.

## Overview

ESBMC PPA location: https://launchpad.net/~esbmc/+archive/ubuntu/esbmc

Currently supported Ubuntu versions:
- **Noble** (24.04)
- **Jammy** (22.04)

## Before You Begin

### Register a Launchpad Account

1. Visit https://launchpad.net/ to register an account
2. Complete account verification
3. Contact the current maintainer (Weiqi Wang <lukewang19@icloud.com>) to request PPA maintenance permissions
4. Wait for the maintainer to approve your permission request

**Note**: You can only upload packages to the PPA after receiving maintenance permissions.

### Prepare Your Working Branch

Before making any changes, you need to work on the `debian-launchpad-patches` branch and merge the latest changes from `master`:

```bash
# Checkout the debian-launchpad-patches branch
git checkout debian-launchpad-patches

# Fetch the latest changes from remote
git fetch origin

# Merge the latest changes from master
git merge origin/master

# If there are conflicts, resolve them and commit
# git add <resolved-files>
# git commit
```

**Important**: Always work on the `debian-launchpad-patches` branch for PPA maintenance. This branch contains all the Debian packaging files and patches.

## Table of Contents

- [Version Management](#version-management)
- [Updating Packages](#updating-packages)
- [Managing Patches](#managing-patches)
- [Building and Uploading](#building-and-uploading)
- [Handling Multiple Ubuntu Versions](#handling-multiple-ubuntu-versions)
- [Common Issues](#common-issues)

## Version Management

### Version Number Format

ESBMC uses the following version number format:

- **Main version**: `7.11.0` (from upstream)
- **Fix version**: `7.11.0+fix1-X` (Debian-specific fixes)
- **Ubuntu version suffix**:
  - Noble: `7.11.0+fix1-X`
  - Jammy: `7.11.0+fix1-X~jammy1`

### Version Comparison Rules

Debian version comparison rules:
- `+` has special meaning in version comparison
- `7.11.0+fix1-9` > `7.11.0-6` (versions with `+` are considered newer)
- `7.11.0+fix1-9` > `7.11.0+fix1-9~jammy1` (no suffix > with suffix)

## Updating Packages

**Important**: Make sure you are on the `debian-launchpad-patches` branch before making any changes:

```bash
git checkout debian-launchpad-patches
git merge origin/master  # Merge latest changes from master
```

### 1. Update changelog

Edit `debian/changelog` and add a new entry at the top:

```bash
dch -i  # Interactive editing
# or
dch -v 7.11.0+fix1-10 "Describe your changes"
```

**Important**:
- Noble version entries must be at the top
- Jammy versions use the `~jammy1` suffix
- Each entry must include a complete description of changes

Example:

```
esbmc (7.11.0+fix1-10) noble; urgency=medium

  * Fix description 1
  * Fix description 2

 -- Your Name <your.email@example.com>  Mon, 01 Jan 2024 12:00:00 +0000

esbmc (7.11.0+fix1-10~jammy1) jammy; urgency=medium

  * Backport to Ubuntu 22.04 Jammy
  * Fix description 1
  * Fix description 2

 -- Your Name <your.email@example.com>  Mon, 01 Jan 2024 12:00:00 +0000
```

### 2. Update Dependencies (if needed)

Edit `debian/control`:

- **Build-Depends**: Dependencies needed at build time
- **Depends**: Dependencies needed at runtime

Example: Adding a new runtime dependency

```
Depends: ${shlibs:Depends}, ${misc:Depends},
         python3,
         python3-mypy,
         libstdc++6,
         libstdc++-dev,
         new-dependency
```

## Managing Patches

### Creating a New Patch

1. Modify the source code
2. Use `dpkg-source` to create a patch:

```bash
# Make sure you're in the source directory
cd /path/to/esbmc

# Commit changes to quilt
dpkg-source --commit

# Enter patch name (e.g., fix-new-issue)
```

3. The patch will be automatically added to `debian/patches/series`

### Modifying an Existing Patch

```bash
# Edit the patch
quilt edit path/to/file.cpp

# Refresh the patch
quilt refresh
```

### Viewing Patch List

```bash
cat debian/patches/series
```

### Applying/Unapplying Patches

```bash
# Apply all patches
quilt push -a

# Unapply all patches
quilt pop -a
```

## Building and Uploading

### 1. Prepare Build Environment

```bash
# Make sure you're in the source root directory
cd /path/to/esbmc

# Clean previous builds
debuild clean
```

### 2. Build Source Package

```bash
# Build source package (including original tarball)
debuild -S -sa

# Or, if the original tarball already exists, don't include it
debuild -S
```

**Note**:
- `-S`: Build source package only
- `-sa`: Include original tarball (needed for first upload)
- Build will generate `.dsc`, `.changes` files in the parent directory

### 3. Check Build Results

```bash
# View generated files
ls -lh ../esbmc_*

# Check package information
dpkg-source -I ../esbmc_*.dsc
```

### 4. Upload to PPA

```bash
# Upload (requires network connection, may need VPN)
dput ppa:esbmc/esbmc ../esbmc_*_source.changes
```

**Network Issues**:
- If upload gets stuck, you may need to use a VPN (e.g., Cloudflare WARP)
- Install WARP: https://pkg.cloudflareclient.com/
- Connect: `warp-cli connect`
- Disconnect after upload: `warp-cli disconnect`

### 5. Check Upload Status

Visit the Launchpad PPA page to check build status:
https://launchpad.net/~esbmc/+archive/ubuntu/esbmc

## Handling Multiple Ubuntu Versions

### Updating Multiple Versions Simultaneously

**Important**: In `debian/changelog`, the Noble version entry must always be at the top (before Jammy), as it has a higher version number (`7.11.0+fix1-X` > `7.11.0+fix1-X~jammy1`).

1. **Prepare changelog with Noble entry first**:
   ```bash
   # Add Noble entry at the top of changelog
   dch -v 7.11.0+fix1-10
   # Edit changelog to add your changes description
   ```

2. **Add Jammy entry after Noble**:
   ```bash
   # Add Jammy entry (it will be added after Noble)
   dch -v 7.11.0+fix1-10~jammy1
   # Edit changelog to add "Backport to Ubuntu 22.04 Jammy" and changes
   ```

3. **Upload Noble version first** (recommended):
   ```bash
   # Build and upload Noble version
   debuild -S -sa
   dput ppa:esbmc/esbmc ../esbmc_7.11.0+fix1-10_source.changes
   ```

4. **Then upload Jammy version**:
   ```bash
   # Build and upload Jammy version
   debuild -S -sa
   dput ppa:esbmc/esbmc ../esbmc_7.11.0+fix1-10~jammy1_source.changes
   ```

**Note**: The upload order is not strictly required (they are different distributions), but uploading Noble first is recommended for logical consistency. The important part is that Noble entry must be at the top of the changelog.

### Version-Specific Fixes

If a fix only applies to a specific Ubuntu version:

1. **Jammy-specific fixes**:
   - Create a patch in `debian/patches/`
   - Include the patch in `debian/patches/series` only for jammy version
   - Use conditional patches (requires modifying `debian/rules`)

2. **Noble-specific fixes**:
   - Similar approach, but typically noble version should include all fixes

## Common Issues

### 1. Version Conflict Error

**Error**: `Version older than that in the archive`

**Cause**: New version number is considered older than existing version

**Solution**:
- Check version number format
- Ensure using `+fix1-X` format instead of `-X` format
- Increment the fix number

### 2. Original Tarball Conflict

**Error**: `File esbmc_7.11.0.orig.tar.gz already exists`

**Cause**: Original tarball already exists but with different content

**Solution**:
- If only Debian-specific changes, use `debuild -S` (without original tarball)
- If upstream version update, need to upload new original tarball

### 3. Patch Application Failure

**Error**: `dpkg-source: error: LC_ALL=C patch ... subprocess returned exit status 1`

**Cause**: Patch doesn't match current source code

**Solution**:
1. Check if patch is correct
2. Ensure source tree is clean
3. Regenerate patch: `dpkg-source --commit`

### 4. GPG Signature Failure

**Error**: `gpg: signing failed`

**Solution**:
- Check if GPG key is correctly configured
- Ensure `gpg-agent` is running
- Check if key has expired

### 5. Upload Timeout

**Error**: Upload gets stuck or times out

**Solution**:
- Use VPN (e.g., Cloudflare WARP)
- Check network connection
- Retry upload

### 6. Build Dependency Issues

**Error**: Missing dependencies during build

**Solution**:
- Check `Build-Depends` in `debian/control`
- Install missing dependencies: `sudo apt build-dep esbmc`

## Useful Commands

```bash
# View current version
dpkg -l | grep esbmc

# View package information
dpkg -I esbmc_*.deb

# View changelog
zcat /usr/share/doc/esbmc/changelog.Debian.gz | less

# Check patch status
quilt series -v

# View build log
cat ../esbmc_*_source.build
```

## Related Resources

- [Debian Packaging Guide](https://www.debian.org/doc/manuals/maint-guide/)
- [Launchpad PPA Documentation](https://help.launchpad.net/Packaging/PPA)
- [ESBMC GitHub](https://github.com/esbmc/esbmc)
- [PPA Page](https://launchpad.net/~esbmc/+archive/ubuntu/esbmc)

---

**Maintainer**: Weiqi Wang <lukewang19@icloud.com>  
**Last Updated**: 2024-12-04
