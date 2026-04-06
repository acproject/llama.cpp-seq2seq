find_package(Threads REQUIRED)

if (NOT DEFINED LLAMA_CPP_SOURCE_DIR OR LLAMA_CPP_SOURCE_DIR STREQUAL "")
    set(LLAMA_CPP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp")
endif()
if (NOT DEFINED LLAMA_CPP_BINARY_DIR OR LLAMA_CPP_BINARY_DIR STREQUAL "")
    set(LLAMA_CPP_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/llama.cpp")
endif()

set(SD_REQUIRED_SOURCES
    sd.cpp
)

foreach(src IN LISTS SD_REQUIRED_SOURCES)
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${src}")
        list(APPEND SD_MISSING_SOURCES "${src}")
    endif()
endforeach()

if(NOT SD_MISSING_SOURCES)
    set(TARGET sd)
    add_library(${TARGET} STATIC ${SD_REQUIRED_SOURCES})
    target_include_directories(${TARGET}
        PUBLIC
            ${CMAKE_CURRENT_LIST_DIR}
        PRIVATE
            ${LLAMA_CPP_SOURCE_DIR}
            ${LLAMA_CPP_SOURCE_DIR}/include
            ${LLAMA_CPP_SOURCE_DIR}/ggml/include
    )
    target_link_libraries(${TARGET}
        PUBLIC
            stable-diffusion
            ggml
            Threads::Threads
    )
    target_compile_features(${TARGET} PUBLIC cxx_std_17)

    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET}
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
        )
    endif()
else()
    message(FATAL_ERROR "Missing sd sources: ${SD_MISSING_SOURCES}")
endif()
