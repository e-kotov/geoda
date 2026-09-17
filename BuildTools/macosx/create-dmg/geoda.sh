#!/bin/sh

VERSION=$1

# In dev builds, package as GeoDa-Metal-dev.app
if [ -d "../build/GeoDa.app" ]; then
    echo "Renaming ../build/GeoDa.app to ../build/GeoDa-Metal-dev.app..."
    mv ../build/GeoDa.app ../build/GeoDa-Metal-dev.app
    /usr/libexec/PlistBuddy -c "Set :CFBundleName GeoDa-Metal-dev" ../build/GeoDa-Metal-dev.app/Contents/Info.plist || true
    /usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName GeoDa Metal Dev" ../build/GeoDa-Metal-dev.app/Contents/Info.plist || true
    /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier edu.uchicago.spatial.metal-dev" ../build/GeoDa-Metal-dev.app/Contents/Info.plist || true
    for f in ../build/GeoDa-Metal-dev.app/Contents/MacOS/*; do
        if [ -f "$f" ]; then
            codesign --force --sign - "$f" 2>/dev/null || true
        fi
    done
    codesign --force --deep --sign - ../build/GeoDa-Metal-dev.app || true
fi

APP_NAME="GeoDa-Metal-dev.app"
if [ ! -d "../build/$APP_NAME" ] && [ -d "../build/GeoDa.app" ]; then
    APP_NAME="GeoDa.app"
fi

./create-dmg --no-internet-enable --volname "GeoDa $VERSION Installer" --volicon "GeoDa_installer.icns" --window-pos 200 120 --window-size 800 400 --icon-size 100 --icon "$APP_NAME" 200 190 --hide-extension "$APP_NAME" --background "geoda_installer_bg.png" --app-drop-link 600 185 GeoDa$VERSION-Installer.dmg ../build/
