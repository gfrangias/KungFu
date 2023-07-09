#!/bin/bash

# Create virtual network device
ip link add kungfu0 type veth peer name kungfu1

# Enable the virtual network device
ip link set kungfu0 up

# Assign IP address and subnet mask
ip addr add 172.16.0.254/24 dev kungfu0

# Verify the configuration
ip addr show kungfu0

# Optional: Test connectivity
ping -c 4 172.16.0.254

echo "Virtual network device 'kungfu0' has been created and configured."
