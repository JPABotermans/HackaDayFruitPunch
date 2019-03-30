@Echo off
:: Check for permissions
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"

:: If error flag set, we do not have admin.
if '%errorlevel%' NEQ '0' (
Echo Requesting administrative privileges...
goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
Echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
Echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"

"%temp%\getadmin.vbs"
Exit /B

:gotAdmin
if exist "%temp%\getadmin.vbs" ( Del "%temp%\getadmin.vbs" )
Pushd "%CD%"
CD /D "%~dp0"
:--------------------------------------

if exist "%ProgramFiles% (x86)\openvpn\config"	(
copy hackathon_31032019_04.crt "%ProgramFiles% (x86)\openvpn\config\"
copy hackathon_31032019_04.key "%ProgramFiles% (x86)\openvpn\config\"
copy psscopenvpn.crt "%ProgramFiles% (x86)\openvpn\config\"
copy hackathon_31032019_04.psscopenvpn.ovpn "%ProgramFiles% (x86)\openvpn\config\"
echo "Files copied to %ProgramFiles% (x86)\openvpn\config\"
)
 
if exist "%ProgramFiles%\openvpn\config"	(
copy hackathon_31032019_04.crt "%ProgramFiles%\openvpn\config\"
copy hackathon_31032019_04.key "%ProgramFiles%\openvpn\config\"
copy psscopenvpn.crt "%ProgramFiles%\openvpn\config\"
copy hackathon_31032019_04.psscopenvpn.ovpn "%ProgramFiles%\openvpn\config\"
echo "Files copied to %ProgramFiles%\openvpn\config\"
)
 
if not exist "%ProgramFiles% (x86)\openvpn\config" (
  if not exist "%ProgramFiles%\openvpn\config" (
    echo MsgBox "OpenVPN Client seems not installed!!"+vbCrLf+" Installation package is available at: http://openvpn.net/index.php/open-source/downloads.html"+vbCrLf+"Once installed, start OpenVPN with Admin rights and re-install the VPN certificate",16,"OpenVPN Client Not found!!!" > "%temp%\vpnmsgbox.vbs"
    "%temp%\vpnmsgbox.vbs"
  )
)
if exist "%temp%\vpnmsgbox.vbs" ( del "%temp%\vpnmsgbox.vbs" )
 
del hackathon_31032019_04.crt
del hackathon_31032019_04.key
del psscopenvpn.crt
del hackathon_31032019_04.psscopenvpn.ovpn
del start_openvpn.sh
del install.bat
