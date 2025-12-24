#!/usr/bin/env bash

echo "ESBMC Linux Install Script"
echo

commands=("wget" "unzip" "uuidgen")

for cmd in "${commands[@]}"; do
    if command -v "$cmd" &> /dev/null; then
        echo "✓ $cmd is installed"
    else
        echo "✗ $cmd is not installed"
        exit 1
    fi
done
echo

TMP_DIR="/tmp/esbmc-web-install-$(uuidgen)"

echo "🚧 Working Directory: $TMP_DIR"

cleanup() {
    if [ -d "$TMP_DIR" ]; then
        echo "🧹☁️ Cleaning up $TMP_DIR..."
        rm -r "$TMP_DIR"
    fi
}

trap cleanup EXIT

if ! mkdir -p "$TMP_DIR"; then
    echo "❌ Failed to make a temporary directory"
    exit 1
fi

download_and_install() {
    cd "$TMP_DIR" || (echo "Failed to cd to $TMP_DIR"; exit 1)
    
    echo "⏳️ Downloading latest version from https://github.com/esbmc/esbmc/releases/latest/download/esbmc-linux.zip"
    if ! wget -q "https://github.com/esbmc/esbmc/releases/latest/download/esbmc-linux.zip"; then
        echo "❌ Failed to download ESBMC from GitHub"
        exit 1
    fi

    echo "⏳️ Extracting esbmc-linux.zip"
    if ! unzip -q "$TMP_DIR/esbmc-linux.zip"; then
        echo "❌ Failed to unzip file in $TMP_DIR/esbmc-linux.zip"
        exit 1
    fi

    if ! [ -f "$TMP_DIR/bin/esbmc" ]; then
        echo "❌ Failed to find ESBMC in $TMP_DIR"
        exit 1
    fi

    if [ -z "$HOME" ]; then
        echo "❌ HOME environment variable is not set"
        exit 1
    fi

    if [ ! -d "$HOME/.local/bin" ]; then
        echo "Creating directory $HOME/.local/bin ..."
        if ! mkdir -p "$HOME/.local/bin"; then
            echo "❌ Failed to create directory $HOME/.local/bin"
            exit 1
        fi
    fi

    if ! mv "$TMP_DIR/bin/esbmc" "$HOME/.local/bin/"; then
        echo "❌ Failed to move ESBMC to location ~/.local/bin"
        exit 1
    fi

    if ! chmod +x "$HOME/.local/bin/esbmc"; then
        echo "❌ Failed to set executable permissions on esbmc"
        exit 1
    fi

    echo
    echo "🌟 $("$HOME/.local/bin/esbmc" --version 2>&1) is successfully installed at ~/.local/bin/esbmc"
    echo

    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo ""
        echo "⚠️ WARNING: $HOME/.local/bin is not in your PATH"
        echo "To use ESBMC, add it to your PATH by adding this line to your shell config:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
        echo "For bash, add to ~/.bashrc"
        echo "For zsh, add to ~/.zshrc"
    fi
}

download_and_install
cleanup
