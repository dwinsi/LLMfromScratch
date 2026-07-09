# Pointing Saathi at a remote Ollama server

By default Saathi talks to Ollama on your own machine
(`http://localhost:11434`). If your laptop is slow, you can run Ollama on a
beefier box (a GPU server, workstation, or cloud VM) and point Saathi at it —
**no code changes, just one setting.**

Saathi reads the server URL from `SAATHI_OLLAMA_BASE_URL` (env var or `.env`).

---

## TL;DR

```bash
# On the SERVER: make Ollama listen on the network, then pull your model
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama pull gemma4:12b

# On your LAPTOP (client): point Saathi at the server
export SAATHI_OLLAMA_BASE_URL=http://SERVER_IP:11434
export SAATHI_OLLAMA_MODEL=gemma4:12b
saathi /doctor        # should show Ollama reachable + model available
```

> ⚠️ Ollama has **no built-in authentication**. Only expose it directly on a
> trusted private network. For anything else use an SSH tunnel or Tailscale
> (see [Securing access](#securing-access)).

---

## 1. Set up the server

On the machine that will run the model:

1. **Install Ollama** (see <https://ollama.com/download>).
2. **Bind to the network.** By default Ollama only listens on `127.0.0.1`. Set
   `OLLAMA_HOST` so it accepts remote connections:

   ```bash
   # Linux (systemd): edit the service
   sudo systemctl edit ollama
   # add:
   #   [Service]
   #   Environment="OLLAMA_HOST=0.0.0.0:11434"
   sudo systemctl restart ollama

   # or, running it manually:
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

3. **Pull the model** you want Saathi to use:

   ```bash
   ollama pull gemma4:12b
   ```

4. **Open the port** (`11434`) in the server firewall for your client only —
   e.g. `sudo ufw allow from <LAPTOP_IP> to any port 11434`.

5. *(Optional, faster)* Let the server keep the model resident and serve
   requests in parallel:

   ```bash
   OLLAMA_KEEP_ALIVE=30m OLLAMA_NUM_PARALLEL=4 ollama serve
   ```

   `OLLAMA_NUM_PARALLEL` is what lets Saathi's parallel tool calls and the
   `/code-review` reviewers actually run concurrently instead of queuing.

---

## 2. Point Saathi at it

On your laptop, set the base URL to the server. Put it in `.env` (recommended)
or export it:

```bash
# .env
SAATHI_OLLAMA_BASE_URL=http://SERVER_IP:11434
SAATHI_OLLAMA_MODEL=gemma4:12b
# a bigger server can afford a bigger context window:
SAATHI_CONTEXT_WINDOW=65536
```

Verify:

```bash
saathi /doctor
```

You should see **Ollama server: reachable** and **Model: available**. Then smoke
test a turn:

```bash
saathi --print "reply with the word PONG"
```

To switch back to local, just unset `SAATHI_OLLAMA_BASE_URL` (or set it back to
`http://localhost:11434`). You can also switch models per session at runtime with
`/model <id>`.

---

## Securing access

Ollama exposes **no authentication**, so don't put a raw `:11434` on the public
internet. Pick one:

### Option A — SSH tunnel (simplest, secure)

Forward the server's Ollama port to your laptop over SSH:

```bash
ssh -N -L 11434:localhost:11434 user@SERVER_IP
```

Now the server's Ollama appears at `http://localhost:11434` on your laptop — so
you leave `SAATHI_OLLAMA_BASE_URL` at its **default** and change nothing else.
On the server, Ollama can stay bound to `127.0.0.1` (no firewall hole needed).

### Option B — Tailscale / VPN (best for "always on")

Put both machines on a [Tailscale](https://tailscale.com) tailnet and use the
server's tailnet IP:

```bash
SAATHI_OLLAMA_BASE_URL=http://100.x.y.z:11434
```

Traffic is encrypted and only reachable inside your tailnet.

### Option C — Reverse proxy with TLS (advanced)

Front Ollama with nginx/Caddy terminating HTTPS. Note a **current Saathi
limitation**: it only sends the base URL — it does **not** attach custom auth
headers (e.g. `Authorization: Bearer …`). So secure the proxy with **mTLS or an
IP allowlist** rather than header/bearer auth. (Basic-auth credentials embedded
in the URL — `https://user:pass@host` — may work via the underlying httpx client
but are not officially supported.)

---

## Troubleshooting

| Symptom | Likely cause / fix |
| --------- | -------------------- |
| `/doctor` → *Ollama server: unreachable* | Server not bound to `0.0.0.0` (still `127.0.0.1`), firewall blocking `11434`, or wrong IP. Test with `curl http://SERVER_IP:11434/api/tags`. |
| `/doctor` → *Model: not pulled* | `ollama pull <model>` on the **server**; make sure `SAATHI_OLLAMA_MODEL` matches a tag the server has. |
| First response is very slow, later ones faster | Cold model load. Set `OLLAMA_KEEP_ALIVE` on the server to keep it resident. |
| Brief "connection refused" right after starting the server | Expected during startup — Saathi retries transient connection errors with backoff (`SAATHI_OLLAMA_MAX_RETRIES`). |
| Parallel tool calls / reviewers feel serialized | Set `OLLAMA_NUM_PARALLEL>1` on the server; a single-slot Ollama processes requests one at a time. |

---

## How it works (for the curious)

Every Ollama call in Saathi goes through `make_llm()`
([`src/saathi/agent/graph.py`](../src/saathi/agent/graph.py)), which builds a
`ChatOllama(base_url=settings.ollama_base_url, …)`. Changing
`SAATHI_OLLAMA_BASE_URL` changes where **all** of it points — the agent, history
summarization (`/compact`), and the `/code-review` reviewers — with no other
changes. See [`config.py`](../src/saathi/config.py) for every tunable.
