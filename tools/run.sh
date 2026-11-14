#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${FLINT_ROOT:-}" ]]; then
    if [[ -d "${APP_ROOT}/../flint/src/flint" ]]; then
        FLINT_ROOT="$(cd "${APP_ROOT}/../flint" && pwd)"
    elif [[ -d "${HOME}/flint/src/flint" ]]; then
        FLINT_ROOT="${HOME}/flint"
    else
        FLINT_ROOT="/home/user/flint"
    fi
fi
cd "$APP_ROOT"

if [[ ! -d "${FLINT_ROOT}/src/flint" ]]; then
    echo "[Robotech] ERROR: Flint Engine not found at ${FLINT_ROOT}. Set FLINT_ROOT=/path/to/flint." >&2
    exit 1
fi

MOJO_BIN=""
resolve_mojo() {
    if [[ -n "${MOJO:-}" && -x "${MOJO}" ]]; then
        MOJO_BIN="${MOJO}"
        return
    fi
    local candidate
    for candidate in \
        "${APP_ROOT}/.pixi/envs/default" \
        "${FLINT_ROOT}/.pixi/envs/default" \
        "${APP_ROOT}/../rollercoaster/.pixi/envs/default" \
        "${HOME}/.pixi/envs/turbomojo"; do
        if [[ -x "${candidate}/bin/mojo" ]]; then
            export MOJO_PREFIX="${candidate}"
            export MODULAR_HOME="${candidate}/share/max"
            export LD_LIBRARY_PATH="${candidate}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            MOJO_BIN="${candidate}/bin/mojo"
            return
        fi
    done
    if command -v mojo >/dev/null 2>&1; then
        MOJO_BIN="$(command -v mojo)"
        return
    fi
    echo "[Robotech] ERROR: Mojo 1.2 compiler not found. Run 'pixi install'." >&2
    exit 1
}

ensure_flint_stamp() {
    if [[ ! -f "${FLINT_ROOT}/src/flint/core/build_stamp.mojo" ]]; then
        cat > "${FLINT_ROOT}/src/flint/core/build_stamp.mojo" <<EOF
"""Generated build provenance stamp."""
comptime BUILD_COMMIT: String = "0.1.0"
comptime BUILD_DIRTY: Bool = False
comptime BUILD_MOJO: String = "1.2.0"
EOF
    fi
}

compile_mojo() {
    local src="$1"
    local out="$2"
    local tmp
    tmp="$(mktemp "/tmp/robotech-link-$(basename "$out")-XXXXXX")"
    rm -f "$tmp" "$out"
    "$MOJO_BIN" build "$src" -o "$tmp" \
        -I "${FLINT_ROOT}/src" -I "${APP_ROOT}/src" \
        -O3 --fp-mode contract=off
    cp "$tmp" "$out"
    chmod +x "$out"
    rm -f "$tmp"
}

cmd_build() {
    resolve_mojo
    ensure_flint_stamp
    mkdir -p "${APP_ROOT}/build"
    local target="${1:-all}"
    if [[ "$target" == "all" || "$target" == "game" ]]; then
        echo "[Robotech] Compiling src/main.mojo -> build/robotech-sim ..."
        compile_mojo "${APP_ROOT}/src/main.mojo" "${APP_ROOT}/build/robotech-sim"
    fi
    if [[ "$target" == "all" || "$target" == "test" ]]; then
        echo "[Robotech] Compiling tests/test_robotech.mojo -> build/robotech-test ..."
        compile_mojo "${APP_ROOT}/tests/test_robotech.mojo" "${APP_ROOT}/build/robotech-test"
    fi
    echo "[Robotech] Build complete."
}

cmd_play() {
    cmd_build game
    exec "${APP_ROOT}/build/robotech-sim" "$@"
}

cmd_test() {
    cmd_build test
    "${APP_ROOT}/build/robotech-test"
}

cmd_shot() {
    cmd_build game
    mkdir -p "${APP_ROOT}/build/shots"
    "${APP_ROOT}/build/robotech-sim" --shots
}

cmd_gate() {
    echo "============================================================"
    echo " ROBOTECH: SDF-1 VERITECH COMBAT SIMULATOR — GATE"
    echo "============================================================"
    cmd_build all
    "${APP_ROOT}/build/robotech-test"
    mkdir -p "${APP_ROOT}/build/shots"
    "${APP_ROOT}/build/robotech-sim" --shots
    echo "============================================================"
    echo " [PASS] ROBOTECH GATE PASSED (Tests + 4 Combat Screenshots)"
    echo "============================================================"
}

subcmd="${1:-play}"
shift || true
case "$subcmd" in
    build) cmd_build "$@" ;;
    play)  cmd_play "$@" ;;
    test)  cmd_test "$@" ;;
    shot)  cmd_shot "$@" ;;
    gate)  cmd_gate "$@" ;;
    *)
        echo "Usage: bash tools/run.sh [play|build|test|shot|gate]"
        ;;
esac
