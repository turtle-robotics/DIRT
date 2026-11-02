#!/usr/bin/env bash
# DIRT display setup for the rover's Raspberry Pi 5 (Ubuntu 24.04 + ROS 2 Jazzy).
#
# Put dirt_display.html, dirt_display_node.py and this script in one folder,
# then run from that folder:   bash setup_dirt_display.sh
#
# What it does:
#   1. Installs rosbridge (the WebSocket bridge the display talks to)
#   2. Copies the display and helper node to ~/dirt_display
#   3. Creates two services that start on boot:
#        dirt-rosbridge      rosbridge on port 9090
#        dirt-display-node   sample logger, Pi stats, optional e-stop gate
#   4. Opens the display full screen in Chromium when the desktop logs in
#   5. Turns off screen blanking
#
# Useful afterwards:
#   systemctl --user status dirt-rosbridge dirt-display-node
#   journalctl --user -u dirt-display-node -f
#   ls ~/dirt_samples
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$HOME/dirt_display"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
GATE="${DIRT_ESTOP_GATE:-false}"   # DIRT_ESTOP_GATE=true bash setup_dirt_display.sh to turn the gate on

for f in dirt_display.html dirt_display_node.py; do
  [[ -f "$HERE/$f" ]] || { echo "Missing $f next to this script." >&2; exit 1; }
done
[[ -f "$ROS_SETUP" ]] || { echo "ROS 2 Jazzy not found at $ROS_SETUP." >&2; exit 1; }

echo "== Installing rosbridge"
sudo apt-get update
sudo apt-get install -y ros-jazzy-rosbridge-suite

echo "== Copying files to $DIR"
mkdir -p "$DIR" "$HOME/dirt_samples"
cp "$HERE/dirt_display.html" "$HERE/dirt_display_node.py" "$DIR/"

# Keep the team's ROS_DOMAIN_ID if one is set in ~/.bashrc
DOMAIN_LINE=""
if grep -q "ROS_DOMAIN_ID" "$HOME/.bashrc" 2>/dev/null; then
  DOMAIN_ID="$(grep -oP 'ROS_DOMAIN_ID=\K[0-9]+' "$HOME/.bashrc" | tail -n1 || true)"
  [[ -n "$DOMAIN_ID" ]] && DOMAIN_LINE="Environment=ROS_DOMAIN_ID=$DOMAIN_ID"
fi

echo "== Creating services"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

cat > "$UNIT_DIR/dirt-rosbridge.service" <<EOF
[Unit]
Description=DIRT rosbridge WebSocket server (port 9090)
After=network-online.target

[Service]
$DOMAIN_LINE
ExecStart=/bin/bash -c 'source $ROS_SETUP && exec ros2 launch rosbridge_server rosbridge_websocket_launch.xml'
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

cat > "$UNIT_DIR/dirt-display-node.service" <<EOF
[Unit]
Description=DIRT display helper node (samples, Pi stats, e-stop gate)
After=dirt-rosbridge.service

[Service]
$DOMAIN_LINE
ExecStart=/bin/bash -c 'source $ROS_SETUP && exec python3 $DIR/dirt_display_node.py --ros-args -p gate_enabled:=$GATE'
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now dirt-rosbridge.service dirt-display-node.service
# keep the services running even when nobody is logged in over SSH
sudo loginctl enable-linger "$USER"

echo "== Setting up kiosk autostart"
BROWSER="$(command -v chromium || command -v chromium-browser || true)"
if [[ -z "$BROWSER" ]]; then
  echo "Chromium not found, installing"
  sudo snap install chromium
  BROWSER="$(command -v chromium)"
fi
mkdir -p "$HOME/.config/autostart"
cat > "$HOME/.config/autostart/dirt-display.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=DIRT Field Display
Exec=$BROWSER --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble --password-store=basic --check-for-update-interval=31536000 file://$DIR/dirt_display.html#live
X-GNOME-Autostart-enabled=true
EOF

echo "== Turning off screen blanking"
if command -v gsettings >/dev/null; then
  gsettings set org.gnome.desktop.session idle-delay 0 || true
  gsettings set org.gnome.desktop.screensaver lock-enabled false || true
fi

echo
echo "Done. Reboot to test:  sudo reboot"
echo "The display opens on the TURTLE start menu and goes live after 10 s."
echo "Samples are saved in $HOME/dirt_samples"
if [[ "$GATE" == "true" ]]; then
  echo "E-stop gate is ON: remap Nav2's velocity output to /cmd_vel_nav."
fi
