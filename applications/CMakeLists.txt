cmake_minimum_required(VERSION 3.14)
project(terrain_shower CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Locate the CORSIKA installation ──────────────────────────────────────────
#
# Search order:
#   1. CORSIKA_INSTALL_PREFIX  – path to the CORSIKA install tree directly
#                                (e.g. /n/home12/jlazar/.local)
#   2. CORSIKA_UPSTREAM_BUILD  – path to the cmake build directory; the
#                                install tree is expected at <build>/install
#   3. Default: /n/home12/jlazar/.local  (mimo host)
#
# Override via environment variable or cmake cache variable of the same name.
#
if(DEFINED ENV{CORSIKA_INSTALL_PREFIX})
    set(_CORSIKA_INSTALL "$ENV{CORSIKA_INSTALL_PREFIX}")
elseif(DEFINED CORSIKA_INSTALL_PREFIX)
    set(_CORSIKA_INSTALL "${CORSIKA_INSTALL_PREFIX}")
elseif(DEFINED ENV{CORSIKA_UPSTREAM_BUILD})
    set(_CORSIKA_INSTALL "$ENV{CORSIKA_UPSTREAM_BUILD}/install")
elseif(DEFINED CORSIKA_UPSTREAM_BUILD)
    set(_CORSIKA_INSTALL "${CORSIKA_UPSTREAM_BUILD}/install")
else()
    set(_CORSIKA_INSTALL "/n/home12/jlazar/corsika/build")
endif()

message(STATUS "Using CORSIKA install at: ${_CORSIKA_INSTALL}")

find_package(corsika CONFIG REQUIRED
    PATHS "${_CORSIKA_INSTALL}"
          "${_CORSIKA_INSTALL}/lib/cmake/corsika"
    NO_DEFAULT_PATH)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(terrain_shower terrain_shower.cpp)
target_link_libraries(terrain_shower PRIVATE CORSIKA8::CORSIKA8)

# macOS: SOPHIA's Fortran routines use large stack frames; increase main-thread
# stack to 256 MB to avoid stack exhaustion for high-energy primaries.
if(APPLE)
    target_link_options(terrain_shower PRIVATE -Wl,-stack_size,0x10000000)
endif()

install(TARGETS terrain_shower DESTINATION bin)
