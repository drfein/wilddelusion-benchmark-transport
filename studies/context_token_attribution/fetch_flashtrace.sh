#!/usr/bin/env bash
set -euo pipefail

commit=e15e117a5fbbe6e8ad6ea6d0f1314fc7835e5784
destination=${1:?usage: fetch_flashtrace.sh DESTINATION}

git clone --filter=blob:none --no-checkout \
  https://github.com/wbopan/flashtrace.git "$destination"
git -C "$destination" sparse-checkout init --cone
git -C "$destination" sparse-checkout set flashtrace
git -C "$destination" checkout "$commit"
test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
printf 'FlashTrace pinned at %s in %s\n' "$commit" "$destination"
