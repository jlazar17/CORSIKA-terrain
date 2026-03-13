cmake_minimum_required(VERSION 3.14)
project(terrain_shower CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Locate the CORSIKA installation ──────────────────────────────────────────
#
# Set the install prefix via the CORSIKA_UPSTREAM_BUILD environment variable
# (pointing to the cmake build directory, not the install sub-directory) or via
# the CORSIKA_UPSTREAM_BUILD cmake cache variable.  The project expects the
# CORSIKA install tree to be at <build_dir>/install.
#
# Example:
#   export CORSIKA_UPSTREAM_BUILD=/path/to/corsika_upstream/build
#   cmake -B build -DCMAKE_BUILD_TYPE=Release ..
#   cmake --build build
#   cmake --install build --prefix /opt/terrain_shower
#
if(DEFINED ENV{CORSIKA_UPSTREAM_BUILD})
    set(_CORSIKA_BUILD "$ENV{CORSIKA_UPSTREAM_BUILD}")
elseif(DEFINED CORSIKA_UPSTREAM_BUILD)
    set(_CORSIKA_BUILD "${CORSIKA_UPSTREAM_BUILD}")
else()
    message(FATAL_ERROR
        "Set CORSIKA_UPSTREAM_BUILD to the CORSIKA cmake build directory.\n"
        "  e.g.  export CORSIKA_UPSTREAM_BUILD=/path/to/corsika_upstream/build\n"
        "  or    cmake -DCORSIKA_UPSTREAM_BUILD=... ..")
endif()

set(_CORSIKA_INSTALL "${_CORSIKA_BUILD}/install")
message(STATUS "Using CORSIKA install at: ${_CORSIKA_INSTALL}")

find_package(corsika CONFIG REQUIRED
    PATHS "${_CORSIKA_INSTALL}/lib/cmake/corsika"
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
