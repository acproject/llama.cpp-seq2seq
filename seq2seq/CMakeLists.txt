set(BEAM_SEARCH_OVERLAY_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/src" CACHE PATH "Beam-search overlay source directory")
option(BEAM_SEARCH_BUILD_LLAMA_OVERLAY "Build a mirrored llama overlay target from beam-search sources" OFF)
option(BEAM_SEARCH_REPLACE_RUNTIME_LLAMA "Replace the built libllama runtime artifact with the beam-search overlay library" ON)
set(BEAM_SEARCH_EXTRA_LLAMA_SOURCES
    "nllb-enc.cpp;nllb-dec.cpp"
    CACHE STRING "Additional beam-search overlay source files to compile into the llama overlay target")

if(NOT TARGET llama)
    message(FATAL_ERROR "beam-search requires the upstream llama target to be available")
endif()

target_sources(llama PRIVATE
    "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}/llama-arch_extend.cpp"
)
target_include_directories(llama PRIVATE
    "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}"
)

function(beam_search_collect_overlay_files out_var)
    set(result)
    if(EXISTS "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}")
        file(GLOB_RECURSE result
            RELATIVE "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}"
            CONFIGURE_DEPENDS
            "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}/*.c"
            "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}/*.cpp"
            "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}/*.h"
            "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}/*.hpp"
        )
    endif()
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

function(beam_search_materialize_overlay out_root out_files)
    if(NOT DEFINED LLAMA_CPP_SOURCE_DIR OR LLAMA_CPP_SOURCE_DIR STREQUAL "")
        message(FATAL_ERROR "LLAMA_CPP_SOURCE_DIR is required for beam-search overlay builds")
    endif()

    set(overlay_root "${CMAKE_CURRENT_BINARY_DIR}/llama-overlay")
    file(MAKE_DIRECTORY "${overlay_root}")
    file(COPY "${LLAMA_CPP_SOURCE_DIR}/src" DESTINATION "${overlay_root}")

    beam_search_collect_overlay_files(overlay_files)
    foreach(rel_path IN LISTS overlay_files)
        get_filename_component(dst_dir "${overlay_root}/src/${rel_path}" DIRECTORY)
        file(MAKE_DIRECTORY "${dst_dir}")
        configure_file(
            "${BEAM_SEARCH_OVERLAY_SOURCE_DIR}/${rel_path}"
            "${overlay_root}/src/${rel_path}"
            COPYONLY
        )
    endforeach()

    set(${out_root} "${overlay_root}" PARENT_SCOPE)
    set(${out_files} "${overlay_files}" PARENT_SCOPE)
endfunction()

function(beam_search_create_llama_overlay target_name)
    beam_search_materialize_overlay(overlay_root overlay_files)

    get_target_property(llama_source_dir llama SOURCE_DIR)
    get_target_property(llama_sources llama SOURCES)
    get_target_property(llama_version llama VERSION)
    get_target_property(llama_soversion llama SOVERSION)
    get_target_property(llama_macho_current_version llama MACHO_CURRENT_VERSION)

    if(NOT llama_sources)
        message(FATAL_ERROR "Unable to read llama target sources")
    endif()

    set(materialized_sources)
    set(upstream_rel_sources)
    foreach(src IN LISTS llama_sources)
        if(IS_ABSOLUTE "${src}")
            set(abs_src "${src}")
        else()
            set(abs_src "${llama_source_dir}/${src}")
        endif()

        file(RELATIVE_PATH rel_src "${llama_source_dir}" "${abs_src}")
        if(rel_src MATCHES "^\\.\\.")
            list(APPEND materialized_sources "${abs_src}")
        else()
            if(rel_src MATCHES "\\.(c|cpp)$")
                list(APPEND upstream_rel_sources "${rel_src}")
            endif()
            list(APPEND materialized_sources "${overlay_root}/src/${rel_src}")
        endif()
    endforeach()

    foreach(rel_path IN LISTS overlay_files)
        if(NOT rel_path MATCHES "\\.(c|cpp)$")
            continue()
        endif()

        list(FIND upstream_rel_sources "${rel_path}" source_idx)
        list(FIND BEAM_SEARCH_EXTRA_LLAMA_SOURCES "${rel_path}" extra_idx)
        if(source_idx EQUAL -1 AND NOT extra_idx EQUAL -1)
            list(APPEND materialized_sources "${overlay_root}/src/${rel_path}")
        endif()
    endforeach()

    add_library(${target_name} ${materialized_sources})
    if(NOT llama_version STREQUAL "llama_version-NOTFOUND")
        set_property(TARGET ${target_name} PROPERTY VERSION "${llama_version}")
    endif()
    if(NOT llama_soversion STREQUAL "llama_soversion-NOTFOUND")
        set_property(TARGET ${target_name} PROPERTY SOVERSION "${llama_soversion}")
    endif()
    if(NOT llama_macho_current_version STREQUAL "llama_macho_current_version-NOTFOUND")
        set_property(TARGET ${target_name} PROPERTY MACHO_CURRENT_VERSION "${llama_macho_current_version}")
    endif()

    target_include_directories(${target_name}
        PRIVATE
            "${overlay_root}/src"
        PUBLIC
            "${LLAMA_CPP_SOURCE_DIR}/include"
    )
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_link_libraries(${target_name} PUBLIC ggml)
    add_dependencies(${target_name} llama)

    if(BUILD_SHARED_LIBS)
        set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(${target_name} PRIVATE LLAMA_BUILD)
        target_compile_definitions(${target_name} PUBLIC LLAMA_SHARED)
    endif()

    if(BEAM_SEARCH_REPLACE_RUNTIME_LLAMA AND BUILD_SHARED_LIBS)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${target_name}>
                $<TARGET_FILE:llama>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${target_name}>
                $<TARGET_FILE_DIR:llama>/${CMAKE_SHARED_LIBRARY_PREFIX}${target_name}${CMAKE_SHARED_LIBRARY_SUFFIX}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${target_name}>
                $<TARGET_FILE_DIR:llama>/${CMAKE_SHARED_LIBRARY_PREFIX}${target_name}${CMAKE_SHARED_LIBRARY_SUFFIX}.${llama_soversion}
        )
    endif()

    if(overlay_files)
        list(JOIN overlay_files ", " overlay_summary)
        message(STATUS "beam-search overlay files: ${overlay_summary}")
    else()
        message(STATUS "beam-search overlay files: none")
    endif()
endfunction()

if(BEAM_SEARCH_BUILD_LLAMA_OVERLAY)
    beam_search_create_llama_overlay(llama-beam)
endif()
