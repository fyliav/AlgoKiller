#!/usr/bin/env python3
"""Recover and apply the trace-proven field-123 outer decryption layer.

The trace proves the final native loop is:

    output[i] = input[i] ^ phase[i & 7]

where ``phase`` is the little-endian byte representation of
``ROR64(0x41bf475d5be9db41, 0x1a)``.  The loop processes the first 0x120c
bytes.  The preceding memcpy copies 0x120d bytes, but the last byte is not
processed or sent through SetByteArrayRegion.

This is an instance-bounded reconstruction.  It recovers the confirmed
outer XOR layer; it does not claim that the resulting field-123 payload is
business plaintext or identify an upstream standard cipher.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


MASK64 = (1 << 64) - 1
ROTATE_INPUT = 0x41BF475D5BE9DB41
ROTATE_RIGHT = 0x1A
PHASE_WORD = 0xFA76D0506FD1D756
PHASE = bytes.fromhex("56 d7 d1 6f 50 d0 76 fa")

SOURCE_COPY_LENGTH = 0x120D
MESSAGE_LENGTH = 0x120C
FIELD_NUMBER = 123
FIELD_OFFSET = 0x6A
FIELD_LENGTH = 0xAAB
FIELD_TAG = bytes.fromhex("da 07")
FIELD_LENGTH_VARINT = bytes.fromhex("ab 15")

EXPECTED_SOURCE_PREFIX = bytes.fromhex(
    "5c d1 bc 18 1d 19 a3 79 84 c7 8a 6f 50 d6 5b dd"
)
EXPECTED_MESSAGE_PREFIX = bytes.fromhex(
    "0a 06 6d 77 4d c9 d5 83 d2 10 5b 00 00 06 2d 27"
)

HEX_DUMP_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+([^|\r\n]+)")
JNI_RE = re.compile(
    r"SetByteArrayRegion\([^,]+,\s*[^,]+,\s*0x0,\s*(0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+)\)"
)
MEMCPY_RE = re.compile(
    r"__memcpy_aarch64_simd\((0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+)\)"
)


@dataclass(frozen=True)
class TraceBuffers:
    source: bytes
    message: bytes
    source_address: int
    message_address: int
    source_line: int
    message_line: int


def ror64(value: int, count: int) -> int:
    """Return the ARM64-style 64-bit rotate-right result."""
    count &= 63
    value &= MASK64
    if count == 0:
        return value
    return ((value >> count) | (value << (64 - count))) & MASK64


def derive_phase() -> bytes:
    """Derive the phase bytes from the rotate observed in the trace."""
    rotated = ror64(ROTATE_INPUT, ROTATE_RIGHT)
    if rotated != PHASE_WORD:
        raise AssertionError(f"unexpected rotate result: {rotated:#x}")
    phase = rotated.to_bytes(8, "little")
    if phase != PHASE:
        raise AssertionError(f"unexpected phase: {phase.hex()}")
    return phase


def xor_at_global_offset(data: bytes, global_offset: int = 0) -> bytes:
    """Apply/reverse the bytewise XOR at a message-relative offset.

    XOR is its own inverse.  ``global_offset`` matters for field-only input:
    field 123 starts at message offset 0x6a, so its first byte uses phase[2].
    """
    if global_offset < 0:
        raise ValueError("global_offset must be non-negative")
    phase = derive_phase()
    return bytes(value ^ phase[(global_offset + index) & 7] for index, value in enumerate(data))


def decrypt_message(source: bytes) -> bytes:
    """Decrypt the 0x120c-byte source buffer into the JNI message."""
    if len(source) < MESSAGE_LENGTH:
        raise ValueError(f"source is {len(source):#x} bytes; need at least {MESSAGE_LENGTH:#x}")
    return xor_at_global_offset(source[:MESSAGE_LENGTH])


def extract_field123(message: bytes) -> bytes:
    """Extract field 123's length-delimited payload from a decoded message."""
    if len(message) < MESSAGE_LENGTH:
        raise ValueError(f"message is {len(message):#x} bytes; need {MESSAGE_LENGTH:#x}")
    header_start = FIELD_OFFSET - len(FIELD_TAG) - len(FIELD_LENGTH_VARINT)
    if message[header_start:FIELD_OFFSET] != FIELD_TAG + FIELD_LENGTH_VARINT:
        actual = message[header_start:FIELD_OFFSET].hex(" ")
        expected = (FIELD_TAG + FIELD_LENGTH_VARINT).hex(" ")
        raise ValueError(f"field 123 header mismatch: got {actual}, expected {expected}")
    payload = message[FIELD_OFFSET:FIELD_OFFSET + FIELD_LENGTH]
    if len(payload) != FIELD_LENGTH:
        raise ValueError("field 123 payload is truncated")
    return payload


def parse_hex_dump_line(line: str) -> tuple[int, bytes] | None:
    match = HEX_DUMP_RE.match(line)
    if not match:
        return None
    tokens = match.group(2).split()
    byte_tokens = [token for token in tokens if re.fullmatch(r"[0-9a-fA-F]{2}", token)]
    if not byte_tokens:
        return None
    return int(match.group(1), 16), bytes.fromhex("".join(byte_tokens))


def collect_hexdump(lines: list[str], start: int, length: int) -> bytes:
    """Collect a contiguous dump beginning at ``start`` from trace lines."""
    chunks: dict[int, bytes] = {}
    for line in lines:
        parsed = parse_hex_dump_line(line)
        if parsed is None:
            continue
        address, chunk = parsed
        if address < start + length and address + len(chunk) > start:
            chunks[address] = chunk
    output = bytearray()
    address = start
    while len(output) < length:
        chunk = chunks.get(address)
        if chunk is None:
            break
        take = min(len(chunk), length - len(output))
        output.extend(chunk[:take])
        address += take
    if len(output) != length:
        raise ValueError(f"hexdump at {start:#x} has {len(output):#x} bytes; expected {length:#x}")
    return bytes(output)


def extract_trace_buffers(trace_path: Path) -> TraceBuffers:
    """Extract the source memcpy and final JNI buffers from a trace file."""
    source: bytes | None = None
    message: bytes | None = None
    source_address = message_address = 0
    source_line = message_line = 0

    # Do not use read_text().splitlines() here: the supplied trace is about
    # 10 GB.  The two relevant dumps are immediately after their marker, so
    # each can be collected while the file is streamed once.
    with trace_path.open("r", encoding="utf-8", errors="replace") as stream:
        line_number = 0
        iterator = enumerate(stream, 1)
        for line_number, line in iterator:
            if source is None:
                match = MEMCPY_RE.search(line)
                if (
                    match
                    and int(match.group(1), 16) == 0xB4000075FAD84A50
                    and int(match.group(2), 16) == 0xB4000075FACFB520
                    and int(match.group(3), 16) == SOURCE_COPY_LENGTH
                ):
                    source_line = line_number
                    source_address = int(match.group(2), 16)
                    source = collect_stream_dump(iterator, source_address, SOURCE_COPY_LENGTH)
                    continue

            if message is None:
                match = JNI_RE.search(line)
                if match and int(match.group(1), 16) == MESSAGE_LENGTH:
                    message_line = line_number
                    message_address = int(match.group(2), 16)
                    message = collect_stream_dump(iterator, message_address, MESSAGE_LENGTH)

            if source is not None and message is not None:
                break

    if source is None:
        raise ValueError(f"did not find the target source memcpy in {trace_path}")
    if message is None:
        raise ValueError(f"did not find the target JNI output in {trace_path}")

    return TraceBuffers(
        source=source,
        message=message,
        source_address=source_address,
        message_address=message_address,
        source_line=source_line,
        message_line=message_line,
    )


def collect_stream_dump(
    lines: object, start: int, length: int
) -> bytes:
    """Collect the next contiguous hexdump from an ``enumerate`` iterator."""
    output = bytearray()
    expected_address = start
    for _line_number, line in lines:  # type: ignore[union-attr]
        parsed = parse_hex_dump_line(line)
        if parsed is None:
            continue
        address, chunk = parsed
        if address != expected_address:
            if output:
                break
            continue
        take = min(len(chunk), length - len(output))
        output.extend(chunk[:take])
        expected_address += take
        if len(output) == length:
            return bytes(output)
    raise ValueError(f"stream hexdump at {start:#x} has {len(output):#x} bytes; expected {length:#x}")


def verify(source: bytes, message: bytes) -> None:
    """Check the trace anchors and the reversible transform."""
    expected = decrypt_message(source)
    if expected != message:
        for index, (left, right) in enumerate(zip(expected, message)):
            if left != right:
                raise AssertionError(
                    f"source/JNI mismatch at offset {index:#x}: expected {left:02x}, got {right:02x}"
                )
        raise AssertionError("source/JNI buffers have different lengths")
    if source[: len(EXPECTED_SOURCE_PREFIX)] != EXPECTED_SOURCE_PREFIX:
        raise AssertionError("source prefix does not match the trace anchor")
    if message[: len(EXPECTED_MESSAGE_PREFIX)] != EXPECTED_MESSAGE_PREFIX:
        raise AssertionError("JNI message prefix does not match the trace anchor")
    if xor_at_global_offset(message) != source[:MESSAGE_LENGTH]:
        raise AssertionError("XOR transform is not reversible for the extracted buffers")
    field = extract_field123(message)
    if len(field) != FIELD_LENGTH:
        raise AssertionError("unexpected field 123 length")


def write_outputs(output_dir: Path, buffers: TraceBuffers) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded = decrypt_message(buffers.source)
    field = extract_field123(decoded)
    field_source = xor_at_global_offset(field, FIELD_OFFSET)
    (output_dir / "source_obfuscated.bin").write_bytes(buffers.source)
    (output_dir / "message_decoded.bin").write_bytes(decoded)
    (output_dir / "field123.bin").write_bytes(field)
    (output_dir / "field123_pre_outer_xor.bin").write_bytes(field_source)
    metadata = {
        "source_line": buffers.source_line,
        "message_line": buffers.message_line,
        "source_address": hex(buffers.source_address),
        "message_address": hex(buffers.message_address),
        "source_copy_length": hex(SOURCE_COPY_LENGTH),
        "message_length": hex(MESSAGE_LENGTH),
        "field_number": FIELD_NUMBER,
        "field_offset": hex(FIELD_OFFSET),
        "field_length": hex(FIELD_LENGTH),
        "rotate_input": hex(ROTATE_INPUT),
        "rotate_right": hex(ROTATE_RIGHT),
        "phase_word": hex(PHASE_WORD),
        "phase": derive_phase().hex(" "),
        "model": "output[i] = input[i] XOR phase[i & 7]",
        "source_is_business_plaintext": False,
        "upstream_standard_cipher_identified": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="trace_12218_main.log to inspect")
    parser.add_argument("--source", type=Path, help="raw source buffer instead of reading --trace")
    parser.add_argument("--output-dir", type=Path, default=Path("work/field123_output"))
    parser.add_argument("--verify-only", action="store_true", help="verify without writing binary outputs")
    args = parser.parse_args(argv)

    if bool(args.trace) == bool(args.source):
        parser.error("provide exactly one of --trace or --source")

    if args.trace:
        buffers = extract_trace_buffers(args.trace)
        verify(buffers.source, buffers.message)
        if not args.verify_only:
            metadata = write_outputs(args.output_dir, buffers)
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
        else:
            print("trace verification passed")
        return 0

    source = args.source.read_bytes()
    message = decrypt_message(source)
    field = extract_field123(message)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "message_decoded.bin").write_bytes(message)
    (args.output_dir / "field123.bin").write_bytes(field)
    (args.output_dir / "field123_pre_outer_xor.bin").write_bytes(
        xor_at_global_offset(field, FIELD_OFFSET)
    )
    print(json.dumps({"phase": derive_phase().hex(" "), "field_length": hex(len(field))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
