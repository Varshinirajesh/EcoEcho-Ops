# Deploying EcoEcho-Ops (Oracle Cloud Always Free, zero-knowledge walkthrough)

This gets the whole stack running 24/7 on a free Oracle Cloud VM, reachable
at `http://<your-vm-ip>`. No domain purchase required to start.

What's exposed publicly vs. not, and why, is explained in
`docker-compose.prod.yml`'s header comment — read that once before starting.

---

## Part 1 — Create the free VM

1. Go to https://www.oracle.com/cloud/free/ and sign up for an Oracle Cloud
   account. You'll need a credit card for identity verification only — the
   "Always Free" tier genuinely never charges it as long as you stay within
   the free-tier shapes below.
2. Once inside the Oracle Cloud Console: **Compute → Instances → Create
   Instance**.
3. Name it anything (e.g. `ecoecho-ops`).
4. Under **Image and shape**, click "Edit":
   - Image: **Canonical Ubuntu 22.04** (or newer)
   - Shape: click "Change shape" → **Ampere** (ARM) → **VM.Standard.A1.Flex**
     → set **4 OCPUs / 24 GB memory** (the max free allowance — use it all,
     it's still free).
5. Under **Add SSH keys**: choose "Generate a key pair for me", then
   **download the private key** (`.key` file) — you cannot get it again
   later. Save it somewhere safe.
6. Leave networking on the default VCN, click **Create**. Wait a few
   minutes for it to reach "Running".
   - If you get a "capacity unavailable" error: this is a known Oracle
     free-tier quirk (the free ARM pool fills up). Just retry — often works
     within a few tries, sometimes faster off-peak.
7. Copy the instance's **public IP address** from the instance details page.

### Open the firewall for web traffic

Oracle blocks ports by default even after the VM's own firewall allows them.
Two places to open, both need to allow port 80 (and 443 for future HTTPS):

1. **Console-side**: Instance details → the VCN link → your subnet →
   **Security Lists** → default security list → **Add Ingress Rules**:
   - Source CIDR: `0.0.0.0/0`, IP Protocol: TCP, Destination Port: `80`
   - Repeat for port `443`
2. **VM-side** (once you're SSH'd in, see below):
   ```bash
   sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
   sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
   sudo netfilter-persistent save
   ```

---

## Part 2 — Connect and install Docker

1. From your own machine, SSH in (replace the path and IP):
   ```bash
   chmod 400 /path/to/your-downloaded-key.key
   ssh -i /path/to/your-downloaded-key.key ubuntu@<your-vm-ip>
   ```
2. Install Docker Engine + Compose plugin:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER
   ```
   Log out and reconnect (`exit`, then SSH in again) so the group change
   takes effect — after that you can run `docker` without `sudo`.
3. Confirm it worked: `docker run hello-world`.

---

## Part 3 — Get the code onto the VM

Simplest path, no GitHub account needed on the VM side — push your repo to
GitHub from your own machine first (ask me to do this if you haven't), then
on the VM:

```bash
git clone <your-repo-url>
cd CLAUDE-WHALE   # or whatever the folder is named
```

If you'd rather not use GitHub, `scp -i your-key.key -r "e:\CLAUDE WHALE" ubuntu@<vm-ip>:~/ecoecho-ops` works directly from your Windows machine instead — ask and I'll give the exact command for your paths.

---

## Part 4 — Configure secrets

```bash
cp .env.example .env
nano .env
```

Change at minimum:
- `MINIO_ROOT_PASSWORD` → a real random password (not `minioadmin`)
- `GRAFANA_ADMIN_PASSWORD` → a real random password (not `admin`)

Generate strong ones with `openssl rand -base64 24` if you want.

Save (`Ctrl+O`, Enter) and exit (`Ctrl+X`).

---

## Part 5 — Bring the stack up

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

First build takes a while (compiling librosa/scikit-learn deps on ARM) —
10-20 minutes is normal on the free tier's CPU. Watch progress with:

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f
```

Once everything shows `Up`/`healthy`, visit `http://<your-vm-ip>` in a
browser — that's your dashboard, publicly reachable, running for real, for
free, all the time.

Before it has real predictions the model needs to exist — if this is a
fresh volume with no prior MLflow runs, run the baseline trainer once
first (see below) so there's a champion model for inference-api to load.

---

## Part 6 — Seed the baseline model (first deploy only)

The dataset and baseline trainer aren't part of the Docker stack (by
design — see CLAUDE.md). Easiest path: run it once on your own machine
(you already have working data/models from local testing), then upload
`models/` and `storage/mlflow/` to the VM the same way you copied the repo
(`scp -r`), instead of re-running training on the VM. Ask me when you're at
this step and I'll give the exact copy command.

---

## Part 7 — Reaching admin UIs (Grafana / MLflow / MinIO) without exposing them

These aren't public (see `docker-compose.prod.yml`'s header) — reach them
through an SSH tunnel instead, from your own machine:

```bash
ssh -i your-key.key -L 3000:localhost:3000 -L 5000:localhost:5000 -L 9001:localhost:9001 ubuntu@<vm-ip>
```

Then open `http://localhost:3000` (Grafana), `http://localhost:5000`
(MLflow), `http://localhost:9001` (MinIO console) on your own machine as
usual — traffic goes through the tunnel, nothing is open to the internet.

---

## Part 8 — Keeping it running

`restart: unless-stopped` is set on every service in
`docker-compose.prod.yml`, so a VM reboot brings everything back
automatically. To pull code changes later:

```bash
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

## Optional later: a real domain + HTTPS

Buy a domain, point its DNS **A record** at your VM's public IP, then edit
`deploy/Caddyfile` — replace `:80` with your domain name and remove the
port. Caddy provisions a free HTTPS certificate automatically on next
restart (`docker compose -f docker-compose.prod.yml restart caddy`).

## Airflow (retraining pipeline)

Airflow is intentionally not part of this deploy (see CLAUDE.md's resource
note — its full stack is heavy). Bring it up only when you want to trigger
a retrain, same as local: `cd airflow && docker compose up -d`, trigger the
DAG, then `docker compose down` it again. A 24GB VM has much more headroom
than the dev laptop that struggled with both stacks at once, but test it
before assuming it's fine to leave running continuously.
