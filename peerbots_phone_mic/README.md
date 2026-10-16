# phone mic + phone speaker support files

Support files for the `--phone-mic` / `--phone-audio` modes in the project's
`main.py`. There is nothing to run in here; the server lives at
`scripts/phone_mic_server.py`.

- `face.html` — the wrapper page the server hands to the phone. It iframes the
  Peerbots face full-screen, captures the mic, and streams 16 kHz / 16-bit mono
  PCM to the server over a websocket. It also plays Kokoro replies back through
  Web Audio when `--phone-audio` is active.
- `tools/cloudflared` — the Cloudflare Tunnel binary, used by `--phone-tunnel`.

## running it

```bash
# phone supplies the mic, kokoro replies play from the handset speaker
python main.py --phone-mic --phone-audio --no-dual-audio

# same, but reachable from any network over https (no usb, no adb)
python main.py --phone-mic --phone-audio --no-dual-audio --phone-tunnel
```

The server listens on `:8443` and prints where to open the page on the phone.

## getting the page onto the phone

Chrome only exposes `getUserMedia` in a secure context, so `http://<lan-ip>:8443`
will silently fail to get the mic. Two ways that work:

- **adb reverse** — the phone sees the server as `localhost`, which counts as
  secure:
  ```bash
  adb reverse tcp:8443 tcp:8443
  # then open http://localhost:8443/ on the phone
  ```
  Works over usb or over wireless debugging. Note that unplugging the cable
  wipes the reverse rule; re-run it after reconnecting.
- **`--phone-tunnel`** — serves the page over `https://` via a temporary
  trycloudflare URL. Open that on the phone from any network. The URL is random
  and changes on every run.

## optional

- `--phone-token <secret>` — require `?token=<secret>` before the page and the
  audio socket are accepted. Worth setting on the tunnel, since the URL is
  otherwise open to anyone who finds it.
- `cert.pem` / `key.pem` and `.peerbots_token` are left over from the earlier
  standalone server and are gitignored. The integrated server does not read
  them.
- `.venv/` is the old standalone environment and is gitignored. It is unused by
  the main project; delete it to reclaim the space.
