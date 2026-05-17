#!/bin/bash
echo "=== Testing Speech Recognition ==="
echo "1. Enabling customer mode..."
rostopic pub -1 /flag std_msgs/String "data: 'customer_reached'"
sleep 2

echo "2. Listening for 30 seconds..."
echo "   Speak now: 'hello robot', 'can I have water', etc."
echo ""
timeout 30 rostopic echo /speech_recognized

echo ""
echo "=== Test Complete ==="
