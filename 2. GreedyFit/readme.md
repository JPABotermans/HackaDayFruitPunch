# DO **NOT** QUIT THE JUPYTER LAB INSTANCE, This will terminate your container and we have no way restarting it without IBM's assistance which can be slow on the weekend. If you do quit your instance **your team will possibly be without the environment for the entire Hackaday** and you will have to train on your own hardware.

# Getting started

## Connecting to the IBM network

To connect to your IBM server you have to first start a vpn connection to their internal network. There should be a folder included for every team member with his/her own openVPN certificate and configuration. Only 1 device can be connected per certificate at a time. You first have to install an openVPN client, an installer for a windows client is included.

To connect after installation of the client follow these steps:

- **Windows**
  1. open *install.bat*
  2. start *openVPN GUI* (through for example the start menu)
  3. right click on the openVPN tray icon
  4. click connect
- **Linux**
  1. running *start_openvpn.sh* as root should start the connection
  2. if this does not work try running ```sudo openvpn --config *.ovpn``` in the directory with your certificate
  3. ask tech support or install Windows
- **MacOS**
  1. IBM recommends installing *tunnelblick*
  2. double click on your *.ovpn file

## connect to your server

**DISCLAIMER: by connecting to this server you agree that it will only be used for purposes related to this Hackaday, other uses (like mining bitcoin) are strictly prohibited.**

- _GreedyFit_
  - Jupyter labs can be found at: http://10.3.8.112:8888
  - SSH access can be requested at the tech support team.