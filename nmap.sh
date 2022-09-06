#!/bin/bash
echo "Instalando privilégios para NMAP."
echo "Necessário superpoderes."

sudo apt install libcap2-bin
sudo setcap cap_net_raw,cap_net_admin,cap_net_bind_service+eip $(which nmap)

echo Feito!
