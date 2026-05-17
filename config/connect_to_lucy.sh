#!/bin/bash
# ------------------------------------------------------------------
# SMART CONNECTION SCRIPT FOR LUCY (HSR) - WiFi Support
# Usage: source connect_to_lucy.sh
# ------------------------------------------------------------------

# 1. Environment check
if [[ "$CONDA_DEFAULT_ENV" != "noetic_env" ]]; then
    echo "WARNING: You are not in 'noetic_env'."
    echo "Run: mamba activate noetic_env"
fi

# 2. Auto-detect IP on robot's subnet (192.168.50.x)
# Try ethernet (ev3) first, then WiFi
MY_IP=$(ip -4 addr show ev3 2>/dev/null | grep -oP '(?<=inet\s)192\.168\.50\.\d+')

if [ -z "$MY_IP" ]; then
    # Not on ethernet, try WiFi (all interfaces)
    MY_IP=$(ip -4 addr | grep -oP '(?<=inet\s)192\.168\.50\.\d+' | head -n 1)
fi

if [ -z "$MY_IP" ]; then
    echo "ERROR: Not connected to 192.168.50.x network"
    echo "Your current IPs:"
    hostname -I
    return 1 2>/dev/null || exit 1
fi

# 3. Set ROS Network Variables
export ROS_MASTER_URI=http://hsrb.local:11311
export ROS_IP=$MY_IP
unset ROS_HOSTNAME  # CRITICAL: Don't set both

# 4. RoboStack Fixes
unset PYTHONPATH

# 5. Verification
echo "✓ Connected to Lucy!"
echo "-------------------------------------"
echo "  Connection: WiFi"
echo "  ROS_MASTER_URI : $ROS_MASTER_URI"
echo "  ROS_IP         : $ROS_IP"
echo "-------------------------------------"

# 6. Test topic subscription
echo "Testing /hsrb/base_scan..."
if timeout 3 rostopic echo /hsrb/base_scan -n 1 &>/dev/null; then
    echo "✓ Can receive laser scan data"
else
    echo "✗ Cannot receive laser scan"
    echo "  Troubleshooting:"
    echo "  1. Disable firewall: sudo ufw disable"
    echo "  2. Check IP: echo \$ROS_IP (should be 192.168.50.5)"
    echo "  3. Verify master: rostopic list"
fi
