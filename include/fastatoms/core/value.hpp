// SPDX-License-Identifier: MIT
#pragma once

/// \file value.hpp
/// \brief A small, order-preserving dynamic value tree (ASE-style `atoms.info`).

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fastatoms {

/// Raised for any malformed or inconsistent input document.
class InputError : public std::runtime_error {
public:
    explicit InputError(const std::string& message) : std::runtime_error(message) {}
};

/// A dynamically typed configuration value: null, bool, int, float, string,
/// list, or map.
///
/// Maps preserve insertion order (they are a vector of key/value pairs), which
/// keeps round-tripped YAML readable and makes iteration deterministic. Config
/// trees are small, so linear key lookup is not a concern.
class Value {
public:
    enum class Type { Null, Bool, Int, Float, String, List, Map };

    using List = std::vector<Value>;
    using Item = std::pair<std::string, Value>;
    using Map = std::vector<Item>;

    Value() = default;
    Value(bool v) : type_(Type::Bool), bool_(v) {}
    Value(std::int64_t v) : type_(Type::Int), int_(v) {}
    Value(int v) : Value(static_cast<std::int64_t>(v)) {}
    Value(double v) : type_(Type::Float), float_(v) {}
    Value(std::string v) : type_(Type::String), string_(std::move(v)) {}
    Value(const char* v) : Value(std::string(v)) {}
    Value(List v) : type_(Type::List), list_(std::move(v)) {}
    Value(Map v) : type_(Type::Map), map_(std::move(v)) {}

    [[nodiscard]] Type type() const noexcept { return type_; }
    [[nodiscard]] std::string_view type_name() const noexcept;

    [[nodiscard]] bool is_null() const noexcept { return type_ == Type::Null; }
    [[nodiscard]] bool is_bool() const noexcept { return type_ == Type::Bool; }
    [[nodiscard]] bool is_int() const noexcept { return type_ == Type::Int; }
    [[nodiscard]] bool is_float() const noexcept { return type_ == Type::Float; }
    [[nodiscard]] bool is_number() const noexcept { return is_int() || is_float(); }
    [[nodiscard]] bool is_string() const noexcept { return type_ == Type::String; }
    [[nodiscard]] bool is_list() const noexcept { return type_ == Type::List; }
    [[nodiscard]] bool is_map() const noexcept { return type_ == Type::Map; }

    /// \name Scalar accessors
    /// Each throws `InputError` on a type mismatch; `as_double` also accepts
    /// integers and `as_int` accepts integral-valued floats.
    /// @{
    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] std::int64_t as_int() const;
    [[nodiscard]] double as_double() const;
    [[nodiscard]] const std::string& as_string() const;
    /// @}

    /// \name Container accessors
    /// @{
    [[nodiscard]] const List& as_list() const;
    [[nodiscard]] const Map& as_map() const;

    /// Number of elements of a list or map; 0 for scalars and null.
    [[nodiscard]] std::size_t size() const noexcept;

    /// Element of a list. \throws InputError if not a list or out of range.
    [[nodiscard]] const Value& at(std::size_t index) const;

    /// Value for a key. \throws InputError if not a map or key is missing.
    [[nodiscard]] const Value& at(std::string_view key) const;

    [[nodiscard]] bool contains(std::string_view key) const noexcept;

    /// Pointer to the value for `key`, or nullptr.
    [[nodiscard]] const Value* find(std::string_view key) const noexcept;

    /// Dotted-path lookup, e.g. `v["simulation.md.timestep"]`.
    ///
    /// Returns a shared null value when any path component is missing, so
    /// chains such as `v["a.b.c"].value_or(1.0)` never throw.
    [[nodiscard]] const Value& operator[](std::string_view path) const noexcept;
    /// @}

    /// \name Defaulted reads
    /// Return `fallback` when this value is null, and throw `InputError` when
    /// it is present but of an incompatible type.
    /// @{
    [[nodiscard]] bool value_or(bool fallback) const;
    [[nodiscard]] std::int64_t value_or(std::int64_t fallback) const;
    [[nodiscard]] int value_or(int fallback) const;
    [[nodiscard]] double value_or(double fallback) const;
    [[nodiscard]] std::string value_or(std::string_view fallback) const;
    /// Overload for string literals: without it, `value_or("x")` would bind
    /// to the bool overload via the pointer-to-bool standard conversion.
    [[nodiscard]] std::string value_or(const char* fallback) const;
    /// @}

    /// Flatten a list of numbers (or a nested list of lists) into doubles.
    /// \throws InputError if any leaf is not a number.
    [[nodiscard]] std::vector<double> as_doubles() const;

    /// A list of strings. \throws InputError if any element is not a string.
    [[nodiscard]] std::vector<std::string> as_strings() const;

    /// \name Mutation (used by the parsers and by the Python bindings)
    /// @{
    void push_back(Value v);
    void insert(std::string key, Value v);
    /// Reference for a key, creating a null entry if absent.
    /// \throws InputError if this value is not a map (and not null).
    Value& operator[](const std::string& key);
    /// @}

    /// Serialise back to YAML.
    [[nodiscard]] std::string to_yaml(int indent = 0) const;

    /// Compact single-line form, used for diagnostics and `repr`.
    [[nodiscard]] std::string to_string() const;

    /// The shared null value returned by failed path lookups.
    [[nodiscard]] static const Value& null() noexcept;

private:
    [[noreturn]] void type_error(std::string_view expected) const;

    /// This value rendered as a single YAML scalar token.
    [[nodiscard]] std::string scalar_text() const;

    /// Inline "[a, b, c]" / "[]" / "{}" form, or an empty string when this
    /// value must be written in block style.
    [[nodiscard]] std::string flow_or_empty() const;

    Type type_{Type::Null};
    bool bool_{false};
    std::int64_t int_{0};
    double float_{0.0};
    std::string string_;
    List list_;
    Map map_;
};

}  // namespace fastatoms
