#!/usr/bin/env bash
# 16 KB page-size compliance check for an APK (Android 15+ requirement for arm64 apps).
#
# Two independent checks:
#  1. Every arm64-v8a ELF .so has all PT_LOAD segments aligned to >= 0x4000 (llvm-readelf).
#  2. Uncompressed .so entries inside the APK sit on 16 KB zip boundaries (zipalign -P 16), when a
#     zipalign new enough to support -P is available.
#
# Usage: scripts/check_16kb_alignment.sh app/build/outputs/apk/debug/app-debug.apk
set -euo pipefail

APK="${1:-app/build/outputs/apk/debug/app-debug.apk}"
[ -f "$APK" ] || { echo "APK not found: $APK"; exit 2; }

SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
READELF="$(command -v llvm-readelf || find "$SDK/ndk" -name llvm-readelf -path '*darwin*' 2>/dev/null | head -1 || true)"
[ -n "$READELF" ] || READELF="$(find "$SDK/ndk" -name llvm-readelf 2>/dev/null | head -1 || true)"
[ -n "$READELF" ] || { echo "llvm-readelf not found (install an NDK)"; exit 2; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
unzip -qo "$APK" 'lib/arm64-v8a/*.so' -d "$TMP" 2>/dev/null || { echo "no arm64-v8a libraries in APK"; exit 2; }

fail=0
echo "== ELF PT_LOAD alignment (arm64-v8a) =="
for so in "$TMP"/lib/arm64-v8a/*.so; do
  name="$(basename "$so")"
  # Column 8 of LOAD rows is p_align (hex).
  aligns="$("$READELF" -l "$so" | awk '$1=="LOAD"{print $NF}' | sort -u | tr '\n' ' ')"
  bad=0
  for a in $aligns; do
    v=$((a))
    if [ "$v" -lt 16384 ]; then bad=1; fi
  done
  if [ $bad -eq 1 ]; then echo "  FAIL  $name  align=[$aligns]"; fail=1; else echo "  ok    $name  align=[$aligns]"; fi
done

echo "== zip entry alignment =="
ZIPALIGN="$(ls -d "$SDK"/build-tools/*/zipalign 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$ZIPALIGN" ] && { "$ZIPALIGN" 2>&1 || true; } | grep -q 'pagesize_kb'; then
  if "$ZIPALIGN" -c -P 16 -v 4 "$APK" >/dev/null; then echo "  ok    zipalign -P 16"; else echo "  FAIL  zipalign -P 16"; fail=1; fi
else
  echo "  skip  (zipalign with -P support not found; AGP 8.3+ aligns uncompressed .so to 16 KB by default)"
fi

if [ $fail -ne 0 ]; then echo "16 KB alignment check FAILED"; exit 1; fi
echo "16 KB alignment check passed"
