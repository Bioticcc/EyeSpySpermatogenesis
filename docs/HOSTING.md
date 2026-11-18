# Hosted annotation page

From 2026-11-16 (decision 028) the staging annotation page runs on a small
rented server so that outside experts can annotate from their own computers.
This workstation keeps the raw TIFFs, does segmentation and training, and
syncs with the server.

```
annotator's browser ──► Cloudflare (email code) ──► tunnel ──► server
                                                             page + labels.sqlite
                                                             hourly backups
this workstation ◄──── scripts/sync.sh (ssh + rsync) ───────┘
TIFFs, segmentation, training, nightly copy of the database
```

## Where things are

- **Live database:** on the server (`/srv/eyespy/workspace/labels.sqlite`).
  Every label goes there as it is made, and the server keeps its own hourly
  backups.
- **This workstation** holds a copy, refreshed by `scripts/sync.sh pull`.
  It does **not** need to be on for people to annotate. It only needs to be on
  when you sync. The copy is read-only: the local page refuses to start on a
  hosted workspace, so no label is made here and then overwritten.
- **Raw TIFFs never leave this workstation.** The server holds only the display
  JPEGs and outlines (about 4 GB), paper snippets, review panels and guide
  pictures.

## How annotators log in

Cloudflare Access sits in front of the page. You list the annotators' email
addresses in Cloudflare. An annotator opens the link, types their email, and
Cloudflare emails them a 6-digit code (a one-time PIN). There are no passwords
to issue or reset. A login lasts as long as the session duration you set (one
month is sensible). Anyone not on the list gets no further than the email
screen. Inside the page they then type their name as before.

## One-time setup

1. **Server.** Chosen 2026-11-16: **AWS Lightsail**, General purpose, 1 GB RAM
   / 40 GB disk or larger, **Ubuntu 24.04** (OS only), dual-stack networking,
   with a static IP attached and the HTTP (80) firewall rule removed (SSH
   only). Upload your SSH public key when creating it. The login is
   `ubuntu@`; the sync tool uses its passwordless sudo. Any similar server
   works (e.g. Hetzner Cloud with `root@`).
2. **Domain.** Any cheap domain, managed in a free Cloudflare account (buy it
   from Cloudflare Registrar, or point an existing domain's nameservers to
   Cloudflare).
3. **Settings file.** Create `hosting.toml` in the repository root (Git-ignored):

   ```toml
   host = "ubuntu@203.0.113.10"   # ssh target: the server's static IP (root@ also works)
   remote_dir = "/srv/eyespy"
   workspace = "outputs/staging-annotation-2026-11-06-01"
   service = "eyespy-annotate"
   service_user = "eyespy"        # unprivileged user that runs the page
   port = 8765
   keep_days = 14                 # server backups older than this are removed once copied here
   update_at = "22:00"            # queued page updates go live at this time...
   timezone = "America/Los_Angeles"  # ...in this zone (decision 029)
   warn_seconds = 300             # countdown shown to anyone online first
   grace_seconds = 60             # then up to this long for their pages to save and log out
   ```

   The last four lines are optional; these are the defaults.

4. **Upload and start.** `scripts/sync.sh setup` uploads the code, the page
   files (about 4 GB, so the first run takes a while) and the database, and
   installs the page as a service that restarts on failure and on reboot. It
   refuses to run if the server already has a database.
5. **Server basics** (over `ssh ubuntu@…`, with `sudo`): `apt update && apt upgrade`,
   `apt install unattended-upgrades`, and `ufw allow OpenSSH && ufw enable`.
   No web ports are opened: the page listens on 127.0.0.1 only and the
   Cloudflare tunnel connects from inside the server.
6. **Tunnel.** In the Cloudflare dashboard, open Zero Trust (free plan, up to
   50 users), then Networks → Tunnels → Create a tunnel (cloudflared). Run the
   install command it shows on the server. Add a public hostname, e.g.
   `staging.<your domain>`, with service `http://localhost:8765`.
7. **Access.** Zero Trust → Access → Applications → Add a self-hosted
   application for that hostname. Login method: One-time PIN. Policy: Allow,
   Include → Emails → the annotators' addresses (and yours). Session duration:
   1 month.
8. **Test** from a phone off the lab network: open the link, get the code, and
   check that the guide and its pictures load, then sign in to the page.

## Day to day

| Task | What to do |
|---|---|
| Add or remove an annotator | Edit the Access policy's email list in Cloudflare. Nothing changes on the server. |
| Check on the page | `scripts/sync.sh status` (running, label counts, disk space) |
| Get fresh labels here (before training, or any time) | `scripts/sync.sh pull`. Safe while people annotate. |
| Add sections, import paper sheets, add a model-review run, or run any command that writes to the workspace | `scripts/sync.sh checkout` (the page goes **off**), run the commands from [ANNOTATION.md](ANNOTATION.md) as usual, then `scripts/sync.sh checkin` (uploads, page back **on**). |
| Change the page's code | Commit, then `scripts/sync.sh deploy`. It goes live at 22:00 (see [Updates](#updates)). |
| Update the guide pictures after a page change | `PYTHON_WITH_PLAYWRIGHT scripts/make_guide_screenshots.py outputs/staging-annotation-2026-11-06-01`, then `scripts/sync.sh push-files` |

**Checkout safety.** `checkin` refuses to upload if the server's database
changed after `checkout`, so a label is never overwritten. If that happens,
`scripts/sync.sh checkin --discard` turns the page back on and pulls the live
copy. Then check out again and repeat the step. `pull` refuses while checked
out, so your local changes are not overwritten either.

**Nightly copy.** To copy the database here every night (this workstation must
be on at that time, and ssh must work without a passphrase prompt, e.g. through
an ssh agent or a dedicated key), add to `crontab -e`:

```
0 2 * * * cd /home/kyle/Desktop/Adam/EyeSpySpermatogenesisRebooted && scripts/sync.sh pull >> outputs/sync.log 2>&1
```

`pull` also removes server backups older than `keep_days` once they are copied
here. Hourly backups use about 190 MB a day, so the 40 GB disk does not fill up.

## Updates

Page code is not put live straight away by default (decision 029). With people
online the countdown starts at 22:00, so the code goes in at about 22:06.

| Command | What happens |
|---|---|
| `scripts/sync.sh deploy` | Uploads the code to a waiting folder on the server. A timer **on the server** puts it live at 22:00 Pacific, so this workstation can be off. Deploying again before then replaces the waiting code. |
| `scripts/sync.sh deploy --now` | Puts it live at once, **only if nobody has the page open**. Otherwise it refuses, says who is online, and leaves the update queued for 22:00. |
| `scripts/sync.sh deploy --now --force` | Puts it live at once even though people are online, after their countdown. |
| `scripts/sync.sh deploy --cancel` | Drops the waiting update. |
| `scripts/sync.sh status` | Shows who is online, the live code's commit, and any waiting update. |

When an update goes in (at 22:00 or with `--now`):

1. If anyone has the page open, a bar at the top of their page counts down
   5 minutes: "The page is being updated at 10:05 PM … You will be logged
   out then."
2. At the deadline their page covers itself ("Updating the page"), stops
   taking new labels, sends every label still waiting in the tab, and logs
   out (writing the usual logout CSV). The server waits up to 60 seconds for
   this.
3. The page stops (a backup is made). Any session still open, such as a tab
   left open on a computer that went to sleep, is ended, and one CSV of all
   labels is written to `exports/` as `labels_<time>_before-update.csv`.
4. The new code goes in and the page starts again. Open tabs reload by
   themselves and show the sign-in screen with "The page was updated. Please
   log in again."

If nobody is online, steps 1 and 2 are skipped and the update takes a few
seconds. If anything fails, the page is started again on the old code and the
update stays queued for the next night. The server's log of each update:
`journalctl -u eyespy-annotate-update`.

A tab that was offline the whole time keeps its unsaved labels (while it stays
open). When it reconnects it asks the person to log in again, and the labels
are saved as soon as the **same name** logs in.

An update while checked out only swaps the code; the page stays off until
`checkin`.

## If something goes wrong

- **The page is down** (annotators see an error): `scripts/sync.sh status`, then
  on the server `systemctl status eyespy-annotate` and
  `journalctl -u eyespy-annotate -n 50`. Labels an annotator made during a short
  outage wait in their browser and are retried every 5 seconds, as long as the
  tab stays open (the browser warns before closing a tab with unsaved labels).
  A restart (`deploy`, `checkin`) does not break open tabs: the page's token
  is kept in `token.txt` in the server's workspace, and a tab with an old
  token fetches the current one and retries instead of dropping the label.
- **Restore a backup:** `checkout`, copy the chosen file from `backups/` over
  `labels.sqlite` here, then `checkin`.
