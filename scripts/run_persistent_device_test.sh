#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <adb-serial> <fully-qualified-test-class>" >&2
  exit 2
fi

device_serial="$1"
test_class="$2"
android_sdk_root="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}"
adb_bin="$android_sdk_root/platform-tools/adb"

if [[ ! -x "$adb_bin" ]]; then
  echo "adb not found at $adb_bin" >&2
  exit 1
fi

./gradlew assembleDebug assembleDebugAndroidTest
"$adb_bin" -s "$device_serial" install -r -t app/build/outputs/apk/debug/app-debug.apk
"$adb_bin" -s "$device_serial" install -r -t app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
"$adb_bin" -s "$device_serial" shell am instrument -w \
  -e class "$test_class" \
  com.example.pdfgemmarag.test/androidx.test.runner.AndroidJUnitRunner
