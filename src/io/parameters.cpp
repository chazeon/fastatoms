// SPDX-License-Identifier: MIT
#include "fastatoms/io/parameters.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "fastatoms/core/units.hpp"

namespace fastatoms {
namespace {

/// Emit a scalar as YAML, quoting anything that could be mis-parsed on the way
/// back in (empty strings, leading/trailing space, YAML indicators, or text
/// that looks like a number or a boolean).
std::string yaml_scalar(const std::string& text) {
    if (text.empty()) {
        return "\"\"";
    }
    static const std::string kSpecials = ":{}[],&*#?|-<>=!%@`\"'\n";
    const bool needs_quotes =
        std::isspace(static_cast<unsigned char>(text.front())) != 0 ||
        std::isspace(static_cast<unsigned char>(text.back())) != 0 ||
        text.find_first_of(kSpecials) != std::string::npos ||
        text == "true" || text == "false" || text == "null" || text == "yes" || text == "no";
    if (!needs_quotes) {
        // A bare token that parses as a number would come back as a number.
        char* end = nullptr;
        std::strtod(text.c_str(), &end);
        if (end != text.c_str() && *end == '\0') {
            return "\"" + text + "\"";
        }
        return text;
    }
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

/// Shortest decimal form that still reads back as the same double, so that
/// 0.8 is written as "0.8" rather than "0.80000000000000004".
std::string format_double(double v) {
    if (std::isnan(v)) {
        return ".nan";
    }
    if (std::isinf(v)) {
        return v > 0 ? ".inf" : "-.inf";
    }

    std::string text;
    for (int precision = 15; precision <= 17; ++precision) {
        std::ostringstream out;
        out << std::setprecision(precision) << v;
        text = out.str();
        if (std::strtod(text.c_str(), nullptr) == v) {
            break;
        }
    }
    // Keep round-tripped floats recognisable as floats.
    if (text.find_first_of(".eEn") == std::string::npos) {
        text += ".0";
    }
    return text;
}

/// Longest line the emitter will produce in flow style before falling back to
/// block style.
constexpr std::size_t kMaxFlowWidth = 96;

}  // namespace

const Value& Value::null() noexcept {
    static const Value shared;
    return shared;
}

std::string_view Value::type_name() const noexcept {
    switch (type_) {
        case Type::Null:   return "null";
        case Type::Bool:   return "bool";
        case Type::Int:    return "int";
        case Type::Float:  return "float";
        case Type::String: return "string";
        case Type::List:   return "list";
        case Type::Map:    return "map";
    }
    return "unknown";
}

void Value::type_error(std::string_view expected) const {
    throw InputError("expected a " + std::string(expected) + " value but found " +
                     std::string(type_name()));
}

bool Value::as_bool() const {
    if (!is_bool()) {
        type_error("boolean");
    }
    return bool_;
}

std::int64_t Value::as_int() const {
    if (is_int()) {
        return int_;
    }
    if (is_float() && float_ == std::floor(float_) && std::isfinite(float_)) {
        return static_cast<std::int64_t>(float_);
    }
    type_error("integer");
}

double Value::as_double() const {
    if (is_float()) {
        return float_;
    }
    if (is_int()) {
        return static_cast<double>(int_);
    }
    type_error("number");
}

const std::string& Value::as_string() const {
    if (!is_string()) {
        type_error("string");
    }
    return string_;
}

const Value::List& Value::as_list() const {
    if (!is_list()) {
        type_error("list");
    }
    return list_;
}

const Value::Map& Value::as_map() const {
    if (!is_map()) {
        type_error("map");
    }
    return map_;
}

std::size_t Value::size() const noexcept {
    if (is_list()) {
        return list_.size();
    }
    if (is_map()) {
        return map_.size();
    }
    return 0;
}

const Value& Value::at(std::size_t index) const {
    if (!is_list()) {
        type_error("list");
    }
    if (index >= list_.size()) {
        throw InputError("list index " + std::to_string(index) + " is out of range (size " +
                         std::to_string(list_.size()) + ")");
    }
    return list_[index];
}

const Value* Value::find(std::string_view key) const noexcept {
    if (!is_map()) {
        return nullptr;
    }
    for (const auto& [name, value] : map_) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

bool Value::contains(std::string_view key) const noexcept { return find(key) != nullptr; }

const Value& Value::at(std::string_view key) const {
    if (!is_map()) {
        type_error("map");
    }
    if (const Value* found = find(key)) {
        return *found;
    }
    throw InputError("missing required key '" + std::string(key) + "'");
}

const Value& Value::operator[](std::string_view path) const noexcept {
    const Value* current = this;
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t dot = path.find('.', start);
        const std::size_t end = dot == std::string_view::npos ? path.size() : dot;
        const std::string_view key = path.substr(start, end - start);
        if (!key.empty()) {
            const Value* next = current->find(key);
            if (next == nullptr) {
                return null();
            }
            current = next;
        }
        if (dot == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return *current;
}

bool Value::value_or(bool fallback) const { return is_null() ? fallback : as_bool(); }

std::int64_t Value::value_or(std::int64_t fallback) const {
    return is_null() ? fallback : as_int();
}

int Value::value_or(int fallback) const {
    return is_null() ? fallback : static_cast<int>(as_int());
}

double Value::value_or(double fallback) const { return is_null() ? fallback : as_double(); }

std::string Value::value_or(std::string_view fallback) const {
    return is_null() ? std::string(fallback) : as_string();
}

std::string Value::value_or(const char* fallback) const {
    return value_or(std::string_view(fallback));
}

std::vector<double> Value::as_doubles() const {
    std::vector<double> out;
    if (is_number()) {
        out.push_back(as_double());
        return out;
    }
    if (!is_list()) {
        type_error("list of numbers");
    }
    for (const Value& item : list_) {
        if (item.is_number()) {
            out.push_back(item.as_double());
        } else if (item.is_list()) {
            const std::vector<double> nested = item.as_doubles();
            out.insert(out.end(), nested.begin(), nested.end());
        } else {
            throw InputError("expected a number in a numeric list but found " +
                             std::string(item.type_name()));
        }
    }
    return out;
}

std::vector<std::string> Value::as_strings() const {
    std::vector<std::string> out;
    if (is_string()) {
        out.push_back(string_);
        return out;
    }
    if (!is_list()) {
        type_error("list of strings");
    }
    out.reserve(list_.size());
    for (const Value& item : list_) {
        out.push_back(item.as_string());
    }
    return out;
}

void Value::push_back(Value v) {
    if (is_null()) {
        type_ = Type::List;
    }
    if (!is_list()) {
        type_error("list");
    }
    list_.push_back(std::move(v));
}

void Value::insert(std::string key, Value v) {
    if (is_null()) {
        type_ = Type::Map;
    }
    if (!is_map()) {
        type_error("map");
    }
    for (auto& [name, existing] : map_) {
        if (name == key) {
            existing = std::move(v);
            return;
        }
    }
    map_.emplace_back(std::move(key), std::move(v));
}

Value& Value::operator[](const std::string& key) {
    if (is_null()) {
        type_ = Type::Map;
    }
    if (!is_map()) {
        type_error("map");
    }
    for (auto& [name, existing] : map_) {
        if (name == key) {
            return existing;
        }
    }
    map_.emplace_back(key, Value{});
    return map_.back().second;
}

namespace {

/// True for a list whose elements are all scalars, i.e. one that can be
/// written inline as [a, b, c].
bool is_flow_list(const Value& v) {
    if (!v.is_list() || v.size() == 0) {
        return false;
    }
    for (const Value& item : v.as_list()) {
        if (item.is_list() || item.is_map()) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::string Value::flow_or_empty() const {
    if (is_list() && list_.empty()) {
        return "[]";
    }
    if (is_map() && map_.empty()) {
        return "{}";
    }
    if (!is_flow_list(*this)) {
        return {};
    }
    std::string out = "[";
    for (std::size_t k = 0; k < list_.size(); ++k) {
        if (k != 0) {
            out += ", ";
        }
        out += list_[k].scalar_text();
        if (out.size() > kMaxFlowWidth) {
            return {};  // too wide; the caller falls back to block style
        }
    }
    out += "]";
    return out;
}

std::string Value::scalar_text() const {
    switch (type_) {
        case Type::Null:   return "null";
        case Type::Bool:   return bool_ ? "true" : "false";
        case Type::Int:    return std::to_string(int_);
        case Type::Float:  return format_double(float_);
        case Type::String: return yaml_scalar(string_);
        default:           return {};
    }
}

std::string Value::to_yaml(int indent) const {
    const std::string pad(static_cast<std::size_t>(indent) * 2, ' ');
    std::ostringstream out;

    switch (type_) {
        case Type::Null:
        case Type::Bool:
        case Type::Int:
        case Type::Float:
        case Type::String:
            return pad + scalar_text() + "\n";

        case Type::List: {
            if (const std::string flow = flow_or_empty(); !flow.empty()) {
                return pad + flow + "\n";
            }
            for (const Value& item : list_) {
                if (const std::string flow = item.flow_or_empty(); !flow.empty()) {
                    out << pad << "- " << flow << "\n";
                } else if (item.is_map()) {
                    // Start the mapping on the dash line. The block is
                    // rendered one level in, and its leading indent is exactly
                    // as wide as "<pad>- ", so overwriting it lines them up.
                    std::string block = item.to_yaml(indent + 1);
                    block.replace(0, static_cast<std::size_t>(indent + 1) * 2, pad + "- ");
                    out << block;
                } else if (item.is_list()) {
                    out << pad << "-\n" << item.to_yaml(indent + 1);
                } else {
                    out << pad << "- " << item.scalar_text() << "\n";
                }
            }
            return out.str();
        }

        case Type::Map: {
            if (map_.empty()) {
                return pad + "{}\n";
            }
            for (const auto& [key, value] : map_) {
                if (value.is_list() || value.is_map()) {
                    if (const std::string flow = value.flow_or_empty(); !flow.empty()) {
                        out << pad << key << ": " << flow << "\n";
                    } else {
                        out << pad << key << ":\n" << value.to_yaml(indent + 1);
                    }
                } else {
                    out << pad << key << ": " << value.scalar_text() << "\n";
                }
            }
            return out.str();
        }
    }
    return {};
}

std::string Value::to_string() const {
    std::ostringstream out;
    switch (type_) {
        case Type::Null:   return "null";
        case Type::Bool:   return bool_ ? "true" : "false";
        case Type::Int:    return std::to_string(int_);
        case Type::Float:  return format_double(float_);
        case Type::String: return "'" + string_ + "'";
        case Type::List: {
            out << "[";
            for (std::size_t k = 0; k < list_.size(); ++k) {
                out << (k ? ", " : "") << list_[k].to_string();
            }
            out << "]";
            return out.str();
        }
        case Type::Map: {
            out << "{";
            for (std::size_t k = 0; k < map_.size(); ++k) {
                out << (k ? ", " : "") << map_[k].first << ": " << map_[k].second.to_string();
            }
            out << "}";
            return out.str();
        }
    }
    return {};
}

Task task_from_string(std::string_view name) {
    std::string key(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(key.begin(), key.end(), '-', '_');

    if (key == "single_point" || key == "singlepoint" || key == "energy" || key == "static") {
        return Task::SinglePoint;
    }
    if (key == "md" || key == "molecular_dynamics" || key == "dynamics") {
        return Task::MolecularDynamics;
    }
    if (key == "optimize" || key == "optimization" || key == "relax" || key == "minimize" ||
        key == "minimization") {
        return Task::Optimization;
    }
    if (key == "phonon" || key == "phonons" || key == "vibrations" || key == "frequencies") {
        return Task::Phonon;
    }
    if (key == "neb" || key == "band" || key == "reaction_path" || key == "saddle") {
        return Task::NEB;
    }
    return Task::Custom;
}

std::string_view task_to_string(Task task) {
    switch (task) {
        case Task::SinglePoint:        return "single_point";
        case Task::MolecularDynamics:  return "md";
        case Task::Optimization:       return "optimize";
        case Task::Phonon:             return "phonon";
        case Task::NEB:                return "neb";
        case Task::Custom:             return "custom";
    }
    return "custom";
}

double MolecularDynamicsParameters::timestep_internal() const { return timestep_fs * units::fs; }

}  // namespace fastatoms
