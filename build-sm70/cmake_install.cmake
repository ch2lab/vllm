# Install script for directory: /data/src/vllm

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

set(CMAKE_INSTALL_LOCAL_ONLY TRUE)
if(CMAKE_INSTALL_COMPONENT STREQUAL "spinloop" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/spinloop.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/spinloop.abi3.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/spinloop.abi3.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/data/src/vllm/build-sm70/spinloop.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/spinloop.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/spinloop.abi3.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/spinloop.abi3.so"
         OLD_RPATH "/data/vllm-dev/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/spinloop.abi3.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "fs_io_C" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/fs_io_C.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/fs_io_C.abi3.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/fs_io_C.abi3.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/data/src/vllm/build-sm70/fs_io_C.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/fs_io_C.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/fs_io_C.abi3.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/fs_io_C.abi3.so"
         OLD_RPATH "/data/vllm-dev/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/fs_io_C.abi3.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cumem_allocator" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/data/src/vllm/build-sm70/cumem_allocator.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so"
         OLD_RPATH "/data/vllm-dev/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/cumem_allocator.abi3.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/src/vllm/.deps/cutlass-build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_C_stable_libtorch" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C_stable_libtorch.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C_stable_libtorch.abi3.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C_stable_libtorch.abi3.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/data/src/vllm/build-sm70/_C_stable_libtorch.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C_stable_libtorch.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C_stable_libtorch.abi3.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C_stable_libtorch.abi3.so"
         OLD_RPATH "/data/vllm-dev/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_C_stable_libtorch.abi3.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_moe_C_stable_libtorch" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C_stable_libtorch.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C_stable_libtorch.abi3.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C_stable_libtorch.abi3.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/data/src/vllm/build-sm70/_moe_C_stable_libtorch.abi3.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C_stable_libtorch.abi3.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C_stable_libtorch.abi3.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C_stable_libtorch.abi3.so"
         OLD_RPATH "/data/vllm-dev/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_moe_C_stable_libtorch.abi3.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_sm70_turbomind_C" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm" TYPE MODULE FILES "/data/src/vllm/build-sm70/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so"
         OLD_RPATH "/data/vllm-dev/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/vllm/_sm70_turbomind_C.cpython-312-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/vllm/third_party/triton_kernels")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "triton_kernels" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/third_party/triton_kernels/" TYPE DIRECTORY FILES "/data/src/vllm/.deps/triton_kernels-src/python/triton_kernels/triton_kernels/" FILES_MATCHING REGEX "/[^/]*\\.py$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "fmha_sm100" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/third_party/fmha_sm100" TYPE FILE FILES
    "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/__init__.py"
    "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/api.py"
    "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/bench_utils.py"
    "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/jit.py"
    "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/sparse.py"
    "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/sparse_fmha_adapter.py"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "fmha_sm100" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/third_party/fmha_sm100/csrc" TYPE DIRECTORY FILES "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/csrc/" REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/\\.git[^/]*$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "fmha_sm100" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/third_party/fmha_sm100/cute" TYPE DIRECTORY FILES "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/cute/" REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/\\.git[^/]*$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "fmha_sm100" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/third_party/fmha_sm100/cutlass/include" TYPE DIRECTORY FILES "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/cutlass/include/" REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/\\.git[^/]*$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "fmha_sm100" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/third_party/fmha_sm100/cutlass/tools/util/include" TYPE DIRECTORY FILES "/data/src/vllm/.deps/fmha_sm100-src/python/fmha_sm100/cutlass/tools/util/include/" REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/\\.git[^/]*$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_flashmla_C" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/third_party/flashmla" TYPE FILE FILES "/data/src/vllm/vllm/third_party/flashmla/flash_mla_interface.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "tml_fa4" OR NOT CMAKE_INSTALL_COMPONENT)
  
  file(GLOB_RECURSE TML_FA4_PY_FILES
    "/data/src/vllm/.deps/tml_fa4-src/flash_attn/cute/*.py")
  foreach(SRC_FILE ${TML_FA4_PY_FILES})
    file(RELATIVE_PATH REL_PATH
      "/data/src/vllm/.deps/tml_fa4-src/flash_attn/cute" ${SRC_FILE})
    set(DST_FILE
      "${CMAKE_INSTALL_PREFIX}/vllm/third_party/tml_fa4/${REL_PATH}")
    get_filename_component(DST_DIR ${DST_FILE} DIRECTORY)
    file(MAKE_DIRECTORY ${DST_DIR})
    file(READ ${SRC_FILE} FILE_CONTENTS)
    string(REPLACE
      "flash_attn.cute"
      "vllm.third_party.tml_fa4"
      FILE_CONTENTS "${FILE_CONTENTS}")
    file(WRITE ${DST_FILE} "${FILE_CONTENTS}")
  endforeach()

endif()

set(CMAKE_INSTALL_LOCAL_ONLY FALSE)
set(OLD_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/vllm/")
if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/src/vllm/build-sm70/vllm-flash-attn/cmake_install.cmake")
endif()

set(CMAKE_INSTALL_PREFIX "${OLD_CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_LOCAL_ONLY TRUE)
if(CMAKE_INSTALL_COMPONENT STREQUAL "_vllm_fa2_C" OR NOT CMAKE_INSTALL_COMPONENT)
  file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_vllm_fa2_C" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn" TYPE DIRECTORY FILES "/data/src/vllm/.deps/vllm-flash-attn-src/vllm_flash_attn/" FILES_MATCHING REGEX "/[^/]*\\.py$" REGEX "/\\_\\_init\\_\\_\\.py$" EXCLUDE REGEX "/flash\\_attn\\_interface\\.py$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_vllm_fa3_C" OR NOT CMAKE_INSTALL_COMPONENT)
  file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_vllm_fa3_C" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn" TYPE DIRECTORY FILES "/data/src/vllm/.deps/vllm-flash-attn-src/vllm_flash_attn/" FILES_MATCHING REGEX "/[^/]*\\.py$" REGEX "/\\_\\_init\\_\\_\\.py$" EXCLUDE REGEX "/flash\\_attn\\_interface\\.py$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "_vllm_fa4_cutedsl_C" OR NOT CMAKE_INSTALL_COMPONENT)
  
    file(GLOB_RECURSE CUTE_PY_FILES "/data/src/vllm/.deps/vllm-flash-attn-src/flash_attn/cute/*.py")
    foreach(SRC_FILE ${CUTE_PY_FILES})
      file(RELATIVE_PATH REL_PATH "/data/src/vllm/.deps/vllm-flash-attn-src/flash_attn/cute" ${SRC_FILE})
      set(DST_FILE "${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn/cute/${REL_PATH}")
      get_filename_component(DST_DIR ${DST_FILE} DIRECTORY)
      file(MAKE_DIRECTORY ${DST_DIR})
      file(READ ${SRC_FILE} FILE_CONTENTS)
      string(REPLACE "flash_attn.cute" "vllm.vllm_flash_attn.cute" FILE_CONTENTS "${FILE_CONTENTS}")
      file(WRITE ${DST_FILE} "${FILE_CONTENTS}")
    endforeach()
  
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/data/src/vllm/build-sm70/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/data/src/vllm/build-sm70/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
