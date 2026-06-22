#!/bin/bash
set -euo pipefail

# Installer: build the Desktop LLM-RAG.app launcher with the project logo.
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGO_PNG="$PROJECT_DIR/LLM-RAG_logo.png"
ASSETS_DIR="$PROJECT_DIR/assets"
ICNS_PATH="$ASSETS_DIR/AppIcon.icns"
APP_NAME="LLM-RAG.app"
APP_PATH="$HOME/Desktop/$APP_NAME"
LEGACY_LAUNCHER="$HOME/Desktop/Launch_Rag_App.command"

if [[ ! -f "$LOGO_PNG" ]]; then
    echo "Error: logo not found at $LOGO_PNG"
    exit 1
fi

if ! command -v sips >/dev/null 2>&1 || ! command -v iconutil >/dev/null 2>&1; then
    echo "Error: macOS tools 'sips' and 'iconutil' are required."
    exit 1
fi

build_icns_from_png() {
    local src_png="$1"
    local out_icns="$2"
    local work_dir iconset_dir

    work_dir="$(mktemp -d)"
    iconset_dir="$work_dir/AppIcon.iconset"
    mkdir -p "$iconset_dir"

    sips -z 16 16     "$src_png" --out "$iconset_dir/icon_16x16.png" >/dev/null
    sips -z 32 32     "$src_png" --out "$iconset_dir/icon_16x16@2x.png" >/dev/null
    sips -z 32 32     "$src_png" --out "$iconset_dir/icon_32x32.png" >/dev/null
    sips -z 64 64     "$src_png" --out "$iconset_dir/icon_32x32@2x.png" >/dev/null
    sips -z 128 128   "$src_png" --out "$iconset_dir/icon_128x128.png" >/dev/null
    sips -z 256 256   "$src_png" --out "$iconset_dir/icon_128x128@2x.png" >/dev/null
    sips -z 256 256   "$src_png" --out "$iconset_dir/icon_256x256.png" >/dev/null
    sips -z 512 512   "$src_png" --out "$iconset_dir/icon_256x256@2x.png" >/dev/null
    sips -z 512 512   "$src_png" --out "$iconset_dir/icon_512x512.png" >/dev/null
    sips -z 1024 1024 "$src_png" --out "$iconset_dir/icon_512x512@2x.png" >/dev/null

    mkdir -p "$(dirname "$out_icns")"
    iconutil -c icns "$iconset_dir" -o "$out_icns"
    rm -rf "$work_dir"
}

echo "Building app icon from LLM-RAG_logo.png…"
build_icns_from_png "$LOGO_PNG" "$ICNS_PATH"

echo "Creating Desktop launcher at: $APP_PATH"
rm -rf "$APP_PATH"
mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources"
cp "$ICNS_PATH" "$APP_PATH/Contents/Resources/AppIcon.icns"

cat > "$APP_PATH/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.llmware.rag-workspace</string>
    <key>CFBundleName</key>
    <string>LLM-RAG</string>
    <key>CFBundleDisplayName</key>
    <string>LLM-RAG</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0-beta</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

cat > "$APP_PATH/Contents/MacOS/start.command" <<EOF
#!/bin/bash
PROJECT_DIR="$PROJECT_DIR"
cd "\$PROJECT_DIR" || {
    osascript -e 'display alert "LLM-RAG" message "Project folder not found."'
    exit 1
}
exec bash "\$PROJECT_DIR/start_app.sh"
EOF
chmod +x "$APP_PATH/Contents/MacOS/start.command"

cat > "$APP_PATH/Contents/MacOS/launcher" <<'EOF'
#!/bin/bash
APP_MACOS="$(cd "$(dirname "$0")" && pwd)"
open -a Terminal "$APP_MACOS/start.command"
EOF

chmod +x "$APP_PATH/Contents/MacOS/launcher"

# Clear quarantine so macOS allows the app to run when launched from Desktop.
xattr -cr "$APP_PATH" 2>/dev/null || true

if [[ -e "$LEGACY_LAUNCHER" ]]; then
    rm -f "$LEGACY_LAUNCHER"
    echo "Removed legacy launcher: $LEGACY_LAUNCHER"
fi

touch "$APP_PATH"
echo "Install complete. Desktop launcher: $APP_PATH"
echo "Icon source: $LOGO_PNG"
echo "Generated icon: $ICNS_PATH"
