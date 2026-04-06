#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "stable-diffusion.h"

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

bool ends_with(const std::string& str, const std::string&  ending);
bool starts_with(const std::string& str, const std::string&  start);
bool contains(const std::string& str, const std::string& substr);

std::string sd_format(const char* fmt, ...);

void replace_all_chars(std::string& str, char target, char replacement);

int round_up_to(int val, int base);

bool file_exists(const std::string& filename);
bool is_directory(const std::string& path);


#endif // UTIL_H
