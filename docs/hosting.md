# Running the season on a server

The season has to be running for three weeks. A laptop can do it, but macOS
sleeps a laptop on lid-close whatever `caffeinate` says, so if the machine
travels, a small always-on host is the better answer.

**A tunnel is not what you want.** Services like playit.gg or ngrok forward
internet traffic *to* a server you are already running, which does not remove
the need for that server. This project also never accepts an inbound
connection: it makes outbound calls to Alpaca and pushes the dashboard to
GitHub. There is nothing for a tunnel to do.

## What it needs

Almost nothing. Python 3.10+, three pure-Python dependencies, and about 200 MB
of disk for the ledger. No database, no web server, no inbound ports. It is
CPU-idle most of the time: the fastest team ticks every 20 seconds.

| Option | Cost | Notes |
|---|---|---|
| **Oracle Cloud Always Free** | free | 4 ARM cores / 24 GB, genuinely always-on. The most capable free tier. |
| **Hetzner CX22** | ~€4/mo | Simplest to set up; EU regions. |
| **DigitalOcean / Vultr** | ~$5/mo | Fine, marginally pricier. |
| **Raspberry Pi at home** | one-off | Works, if your power and internet are reliable. |
| **An old laptop, lid open** | free | Genuinely fine. Disable sleep. |

Any x86 or ARM Linux with Python 3.10+ works. 1 GB of RAM is plenty.

## Setup

```bash
# 1. Python and git
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# 2. The project
git clone https://github.com/faarisaahmed/trading-competition.git
cd trading-competition
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Credentials -- paste the three key pairs
comp setup-accounts --template > keys.txt
nano keys.txt
comp setup-accounts --from-file keys.txt && rm keys.txt

# 4. Pre-flight
comp doctor --check-accounts        # must say "ready to run"
```

### The clock

The schedule is computed from the *market* calendar, not the server's locale,
so the box's timezone does not affect when rounds start. But its clock must be
*correct*, or a tick can be requested for a bar that does not exist yet:

```bash
timedatectl set-ntp true
timedatectl status | grep -E "synchronized|NTP"
```

### The RL model

Its trained state lives in the ledger, not in git, so it does not travel with
a clone. Either retrain on the server:

```bash
comp pretrain                       # strategy + picker, ~25 minutes
```

or copy `runs/competition.sqlite` across from the machine that trained it.
Retraining is deterministic given the same history, so both give the same
model.

### Publishing the dashboard

The server pushes to the `gh-pages` branch, which needs write access. A
fine-grained personal access token scoped to this one repository with
**Contents: read and write** is the least privilege that works:

```bash
git remote set-url origin https://USERNAME:TOKEN@github.com/USERNAME/trading-competition.git
git config --global credential.helper store
```

The publisher never prints the remote URL in an error, precisely because it
can carry a token. If you would rather not put one on the box, run with
`comp service install --no-publish` and publish from elsewhere.

## Start it

```bash
comp service install
```

On Linux that writes a **systemd user unit**, enables it, and turns on
lingering -- without which a user service stops the moment your SSH session
closes, which is a deeply confusing way to lose a competition.

```bash
comp service status      # active? which pid?
comp service logs        # recent output
journalctl --user -u trading-competition -f    # follow it live
comp service uninstall   # stop and remove
```

It restarts itself if it exits. That is safe rather than merely survivable:
the engine checkpoints every tick, so a restart rejoins the round it was in
rather than starting it over.

## Moving a season that has already started

The ledger is the competition's memory -- results, equity curves, learned
state, and the Round 3 draft. Stop the service, copy it, start the service:

```bash
comp service uninstall                       # on the old machine
scp runs/competition.sqlite user@host:~/trading-competition/runs/
comp service install                         # on the new one
```

Copy the whole `runs/` directory if you want the equity history intact.
Because the engine resumes from a checkpoint, a round that was mid-flight
continues on the new host rather than restarting.
