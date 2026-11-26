"""ASE-style structure I/O on top of the C++ core.

``fastatoms.io.read`` and ``fastatoms.io.write`` mirror ``ase.io.read`` and
``ase.io.write``, so ported code only changes its import. The parsing is Python;
the formats are the C++ readers and writers in ``fastatoms._core``.
"""

from __future__ import annotations

import gzip
import operator
import os
from collections.abc import Iterator
from typing import Any, overload

from fastatoms import _core
from fastatoms._core import Atoms

__all__ = ["iread", "read", "read_string", "string2index", "write", "write_string"]

# Formats whose writer stores several structures in one file, so a list of
# Atoms is valid for them. Every other format holds one structure.
_MULTI_FRAME_WRITE = ("extxyz", "xyz")

# Bytes read to decide a format from content when the name gives nothing.
_SNIFF_BYTES = 4096

_GZIP_MAGIC = b"\x1f\x8b"


def string2index(string: str) -> int | slice:
    """Parse an ASE index string: ``"-1"`` is an int, ``"1:5"`` or ``"::2"`` a slice.

    Follows ``ase.utils.string2index``. Unlike ASE, a string that is neither an
    int nor a slice raises ``ValueError`` instead of returning the string.
    """
    text = string.strip()
    if ":" not in text:
        return int(text)
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"index syntax error: {string!r}")
    fields = [None if part.strip() == "" else int(part) for part in parts]
    if len(fields) == 3 and fields[2] == 0:
        raise ValueError(f"index step cannot be zero: {string!r}")
    return slice(*fields)


def _decode(raw: bytes) -> str:
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    # utf-8-sig drops a leading byte order mark.
    return raw.decode("utf-8-sig")


def _strip_gz(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


def _load(source: Any) -> tuple[str, str | None]:
    """File contents as text, and the name used for format guessing (or None)."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        opener = gzip.open if path.lower().endswith(".gz") else open
        with opener(path, "rb") as handle:
            return _decode(handle.read()), path
    if hasattr(source, "read"):
        data = source.read()
        name = getattr(source, "name", None)
        name = name if isinstance(name, str) else None
        if isinstance(data, bytes):
            return _decode(data), name
        data = data.removeprefix("\ufeff")
        return data, name
    raise TypeError(f"expected a path or a file object, got {type(source).__name__}")


def _guess(name: str | None, text: str, format: str | None) -> str:
    """The format name: the explicit one, else from the name, else from the content."""
    if format is not None:
        return format
    base = _strip_gz(name) if name else ""
    fmt = _core.guess_format(base, "")
    if not fmt:
        head = text.encode("utf-8")[:_SNIFF_BYTES].decode("utf-8", "ignore")
        fmt = _core.guess_format(base, head)
    if not fmt:
        label = name if name else "<file object>"
        raise ValueError(f"cannot determine the file format of {label!r}; pass format=")
    return fmt


def _select(frames: list[Atoms], index: Any) -> Atoms | list[Atoms]:
    """Pick frames the way ASE does: an int gives one, a slice or ':' gives a list."""
    if isinstance(index, str):
        index = string2index(index)
    if isinstance(index, slice):
        return list(frames[index])
    return frames[operator.index(index)]


def _frames(text: str, fmt: str, index: Any, kwargs: dict[str, Any]) -> Atoms | list[Atoms]:
    if index is None:
        # The last frame. The core reads this directly, which is cheaper for trajectories.
        return _core.read(text, fmt, **kwargs)
    if isinstance(index, str):
        index = string2index(index)
    if isinstance(index, slice):
        return list(_core.read_all(text, fmt, **kwargs)[index])
    return _select(_core.read_all(text, fmt, **kwargs), index)


@overload
def read_string(text: str, format: str, index: int | None = None, **kwargs: Any) -> Atoms: ...
@overload
def read_string(text: str, format: str, index: slice | str, **kwargs: Any) -> list[Atoms]: ...
def read_string(text: str, format: str, index: Any = None, **kwargs: Any) -> Atoms | list[Atoms]:
    """Read structures from file contents in ``format``. Same ``index`` rules as ``read``."""
    if isinstance(text, bytes):
        text = _decode(text)
    return _frames(text, format, index, kwargs)


@overload
def read(filename: Any, index: int | None = None, format: str | None = None, **kwargs: Any) -> Atoms: ...
@overload
def read(filename: Any, index: slice | str, format: str | None = None, **kwargs: Any) -> list[Atoms]: ...
def read(filename: Any, index: Any = None, format: str | None = None, **kwargs: Any) -> Atoms | list[Atoms]:
    """Read one structure, or a list of them, like ``ase.io.read``.

    ``filename`` is a path or a file object (text or binary). ``.gz`` paths are
    decompressed. ``index`` is None (the last frame), an int, or a slice or a
    string such as ``":"``, ``"-3:"`` or ``"1:5"``, which gives a list. Keyword
    arguments are the format options, e.g. ``species=`` or ``disorder=``.
    """
    text, name = _load(filename)
    fmt = _guess(name, text, format)
    return _frames(text, fmt, index, kwargs)


def iread(filename: Any, index: Any = None, format: str | None = None, **kwargs: Any) -> Iterator[Atoms]:
    """Yield the frames selected by ``index``. Reads the whole file first (simple, not lazy)."""
    selected = read(filename, index=index, format=format, **kwargs)
    if isinstance(selected, list):
        yield from selected
    else:
        yield selected


def _as_frames(images: Any) -> tuple[list[Atoms], bool]:
    """(frames, given_as_list). ASE Atoms objects are converted on the way in."""
    as_list = isinstance(images, (list, tuple))
    items = list(images) if as_list else [images]
    frames = []
    for item in items:
        if not isinstance(item, Atoms):
            if type(item).__module__.split(".")[0] != "ase":
                raise TypeError(f"cannot write {type(item).__name__}; expected Atoms")
            item = Atoms.from_ase(item)
        frames.append(item)
    return frames, as_list


def _serialise(frames: list[Atoms], as_list: bool, fmt: str, kwargs: dict[str, Any]) -> str:
    if not frames:
        raise ValueError("no structures to write")
    if len(frames) > 1 and fmt not in _MULTI_FRAME_WRITE:
        raise ValueError(
            f"{fmt}-format can only store 1 Atoms object; use a multi-frame format "
            f"({', '.join(_MULTI_FRAME_WRITE)}) to write {len(frames)}"
        )
    if not as_list:
        return _core.write(frames[0], fmt, **kwargs)
    return "".join(_core.write(frame, fmt, **kwargs) for frame in frames)


def write_string(images: Any, format: str, **kwargs: Any) -> str:
    """Serialise one Atoms, or a list for a multi-frame format, to a string."""
    if format is None:
        raise ValueError("write_string needs a format name")
    frames, as_list = _as_frames(images)
    return _serialise(frames, as_list, format, kwargs)


def write(filename: Any, images: Any, format: str | None = None, **kwargs: Any) -> None:
    """Write one Atoms, or a list of them, like ``ase.io.write``.

    ``filename`` is a path or a text file object. A list is written as several
    frames for extxyz/xyz; other formats take one structure. The format is
    guessed from the file name when not given. ``.gz`` paths are compressed.
    """
    frames, as_list = _as_frames(images)
    if isinstance(filename, (str, os.PathLike)):
        name = os.fspath(filename)
    else:
        name = getattr(filename, "name", None)
        name = name if isinstance(name, str) else None
    fmt = _guess(name, "", format)
    text = _serialise(frames, as_list, fmt, kwargs)

    if isinstance(filename, (str, os.PathLike)):
        if name.lower().endswith(".gz"):
            with gzip.open(name, "wt", encoding="utf-8") as handle:
                handle.write(text)
        else:
            with open(name, "w", encoding="utf-8") as handle:
                handle.write(text)
        return
    try:
        filename.write(text)
    except TypeError:
        # A binary file object.
        filename.write(text.encode("utf-8"))
