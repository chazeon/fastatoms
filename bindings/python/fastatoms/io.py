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

# Bytes read to decide a format from content when the name gives nothing.
_SNIFF_BYTES = 4096

_GZIP_MAGIC = b"\x1f\x8b"
_UTF8_BOM = b"\xef\xbb\xbf"


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


def _gunzip(raw: bytes) -> bytes:
    return gzip.decompress(raw) if raw[:2] == _GZIP_MAGIC else raw


def _text(raw: bytes | str) -> str:
    """Contents as text. UTF-8 (BOM dropped) first; Latin-1 when that fails.

    ASE reads CIF as Latin-1 (the IUCr spec requires it). UTF-8 is tried first
    here so valid UTF-8 keeps its non-ASCII characters; Latin-1 is the fallback.
    """
    if isinstance(raw, str):
        return raw.removeprefix("﻿")
    raw = raw.removeprefix(_UTF8_BOM)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _head(raw: bytes | str) -> str:
    """The first bytes of the contents, as text, for format sniffing."""
    if isinstance(raw, str):
        return raw[:_SNIFF_BYTES].removeprefix("﻿")
    return raw[:_SNIFF_BYTES].removeprefix(_UTF8_BOM).decode("utf-8", "ignore")


def _strip_gz(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


def _load(source: Any) -> tuple[bytes | str, str | None]:
    """Decompressed contents (bytes or text) and the file name (or None)."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with open(path, "rb") as handle:
            raw: bytes | str = handle.read()
        name: str | None = path
    elif hasattr(source, "read"):
        raw = source.read()
        name = getattr(source, "name", None)
        name = name if isinstance(name, str) else None
    else:
        raise TypeError(f"expected a path or a file object, got {type(source).__name__}")
    if isinstance(raw, bytes):
        raw = _gunzip(raw)
    return raw, name


def _split_at_sign(filename: Any, index: Any, do_not_split_by_at_sign: bool) -> tuple[Any, Any]:
    """Split ``"traj.xyz@::2"`` into the path and the index, as ``ase.io.formats.parse_filename``.

    The split happens at the last ``@`` of the file name, and only when the
    suffix parses as an index. An explicit slice ``index`` wins over the suffix.
    """
    if not isinstance(filename, (str, os.PathLike)):
        return filename, index
    path = os.fspath(filename)
    if isinstance(index, str):
        index = string2index(index)
    if do_not_split_by_at_sign or "@" not in os.path.basename(path):
        return path, index
    base, _, suffix = path.rpartition("@")
    if isinstance(index, slice):
        return base, index
    try:
        parsed = string2index(suffix)
    except ValueError:
        return path, index
    return base, parsed


def _guess(name: str | None, head: str, format: str | None) -> str:
    """The format name: the explicit one, else the core's guess from name and head."""
    if format is not None:
        return format
    fmt = _core.guess_format(_strip_gz(name) if name else "", head)
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
def read_string(text: str | bytes, format: str, index: int | None = None, **kwargs: Any) -> Atoms: ...
@overload
def read_string(text: str | bytes, format: str, index: slice, **kwargs: Any) -> list[Atoms]: ...
@overload
def read_string(text: str | bytes, format: str, index: str, **kwargs: Any) -> Atoms | list[Atoms]: ...
def read_string(text: str | bytes, format: str, index: Any = None, **kwargs: Any) -> Atoms | list[Atoms]:
    """Read structures from file contents in ``format``. Same ``index`` rules as ``read``."""
    if isinstance(text, bytes):
        text = _text(_gunzip(text))
    return _frames(text, format, index, kwargs)


def _read_resolved(filename: Any, index: Any, format: str | None, kwargs: dict[str, Any]) -> Atoms | list[Atoms]:
    """Read ``filename`` with an already-split path and index."""
    data, name = _load(filename)
    fmt = _guess(name, _head(data), format)
    return _frames(_text(data), fmt, index, kwargs)


@overload
def read(
    filename: Any,
    index: None = None,
    format: str | None = None,
    *,
    do_not_split_by_at_sign: bool = False,
    **kwargs: Any,
) -> Atoms: ...
@overload
def read(
    filename: Any, index: int, format: str | None = None, *, do_not_split_by_at_sign: bool = False, **kwargs: Any
) -> Atoms: ...
@overload
def read(
    filename: Any, index: slice, format: str | None = None, *, do_not_split_by_at_sign: bool = False, **kwargs: Any
) -> list[Atoms]: ...
@overload
def read(
    filename: Any, index: str, format: str | None = None, *, do_not_split_by_at_sign: bool = False, **kwargs: Any
) -> Atoms | list[Atoms]: ...
def read(
    filename: Any,
    index: Any = None,
    format: str | None = None,
    *,
    do_not_split_by_at_sign: bool = False,
    **kwargs: Any,
) -> Atoms | list[Atoms]:
    """Read one structure, or a list of them, like ``ase.io.read``.

    ``filename`` is a path or a file object (text or binary). ``.gz`` contents
    are decompressed. ``index`` is None (the last frame), an int, or a slice or a
    string such as ``":"``, ``"-3:"`` or ``"1:5"``, which gives a list. A path can
    carry the index as a suffix, ``"traj.xyz@::2"``, unless
    ``do_not_split_by_at_sign=True``. Keyword arguments are the format options,
    e.g. ``species=`` or ``disorder=``.
    """
    filename, index = _split_at_sign(filename, index, do_not_split_by_at_sign)
    return _read_resolved(filename, index, format, kwargs)


def iread(
    filename: Any,
    index: Any = None,
    format: str | None = None,
    *,
    do_not_split_by_at_sign: bool = False,
    **kwargs: Any,
) -> Iterator[Atoms]:
    """Yield the frames selected by ``index``; None (the default) yields every frame.

    Reads the whole file first (simple, not lazy).
    """
    filename, index = _split_at_sign(filename, index, do_not_split_by_at_sign)
    if index is None:
        index = slice(None)
    selected = _read_resolved(filename, index, format, kwargs)
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
    if not as_list:
        return _core.write(frames[0], fmt, **kwargs)
    if len(frames) > 1 and not _core.can_write_frames(fmt):
        raise ValueError(
            f"{fmt}-format can only store 1 Atoms object; use a multi-frame format "
            f"(xyz, extxyz, cif) to write {len(frames)}"
        )
    return _core.write_all(frames, fmt, **kwargs)


def write_string(images: Any, format: str, **kwargs: Any) -> str:
    """Serialise one Atoms, or a list for a multi-frame format, to a string."""
    if format is None:
        raise ValueError("write_string needs a format name")
    frames, as_list = _as_frames(images)
    return _serialise(frames, as_list, format, kwargs)


def write(filename: Any, images: Any, format: str | None = None, **kwargs: Any) -> None:
    """Write one Atoms, or a list of them, like ``ase.io.write``.

    ``filename`` is a path or a text file object. A list is written as several
    frames for xyz, extxyz and cif (one data block per structure for cif); other
    formats take one structure. The format is guessed from the file name when not
    given. ``.gz`` paths are compressed.
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
