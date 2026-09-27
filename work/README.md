# Protobuf field 123 outer decryption

`field123_decrypt.py` is the trace-grounded source delivery for
`trace_12218_main.log`.

## Reproduce from the trace

```bash
python3 work/field123_decrypt.py \
  --trace /Users/lidongyooo/custom/google_messages/rcs_traces_ins_notsuccess/trace_12218_main.log \
  --output-dir work/field123_output
```

The command verifies the source memcpy against the final
`SetByteArrayRegion` buffer and writes:

- `source_obfuscated.bin`: the 0x120d-byte pre-final-XOR source copied by native code;
- `message_decoded.bin`: the 0x120c-byte JNI message after the verified XOR layer;
- `field123.bin`: protobuf field 123 payload, offset 0x6a, length 0xaab;
- `field123_pre_outer_xor.bin`: the field bytes before the final outer XOR;
- `metadata.json`: trace addresses, line numbers, constants, and evidence limits.

## Algorithm

The confirmed final loop is:

```text
phase_word = ROR64(0x41bf475d5be9db41, 0x1a)
phase      = phase_word.to_bytes(8, "little")
            = 56 d7 d1 6f 50 d0 76 fa
output[i]  = input[i] XOR phase[i & 7]
```

It processes `0 <= i < 0x120c`; the preceding copy length is `0x120d`.
For field-only input, use the message-relative offset `0x6a`, so the first
field byte uses `phase[0x6a & 7]`.

This recovers the confirmed outer decryption/obfuscation layer. The trace
does not close the upstream producer of the source buffer, so this delivery
does not label field 123's resulting high-entropy payload as business
plaintext and does not claim AES, SM4, or another standard cipher.
