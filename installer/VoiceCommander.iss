#define AppName "VoiceCommander"
#define AppVersion "0.1.0"
#ifndef SourceDir
  #define SourceDir "..\dist\standard\VoiceCommander"
#endif
#ifndef Edition
  #define Edition "standard"
#endif
#if Edition == "nvidia"
  #define EditionName "NVIDIA"
#else
  #define EditionName "Standard"
#endif

[Setup]
AppId={{A81046C0-E035-4B10-92EF-0CCAF4227CC4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion} ({#EditionName})
UninstallDisplayName={#AppName} ({#EditionName})
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist\installer
OutputBaseFilename=VoiceCommander-{#Edition}-Setup-x64
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\VoiceCommander"; Filename: "{app}\VoiceCommander.exe"; WorkingDir: "{app}"
Name: "{group}\VoiceCommander Settings"; Filename: "{app}\VoiceCommander.exe"; Parameters: "--settings"; WorkingDir: "{app}"
Name: "{group}\Uninstall VoiceCommander"; Filename: "{uninstallexe}"
Name: "{userstartup}\VoiceCommander"; Filename: "{app}\VoiceCommander.exe"; WorkingDir: "{app}"

[Run]
Filename: "{app}\VoiceCommander.exe"; Description: "Launch VoiceCommander"; Flags: nowait postinstall skipifsilent
